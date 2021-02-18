#![allow(unused)]

use blake2::{Blake2s, Digest};
use std::borrow::Cow;
use std::collections::HashSet;
use std::future::Future;
use std::num::NonZeroUsize;
use std::task::Poll;
use tokio::io::{AsyncBufRead, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

mod matcher;
use matcher::{CaseInsensitiveASCII, Comparator, Natural};

mod parsing;

#[macro_use]
mod framing;
use framing::{fold, FramingError, ParseResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;
        tokio::spawn(connection(socket));
    }
}

async fn connection<RW>(socket: RW)
where
    RW: AsyncRead + AsyncWrite + Unpin,
{
    let (reader, mut writer) = tokio::io::split(socket);
    let mut reader = BufReader::new(reader);
    loop {
        if let Err(e) = message(&mut reader, &mut writer).await {
            match e {
                FramingError::End => {}
                FramingError::BadSyntax => {
                    let _ = send_response(&mut writer, ()).await;
                }
                FramingError::BadVersion => {
                    let _ = send_response(&mut writer, ()).await;
                }
            }
            break;
        }
    }
}

async fn skip(mut amt: usize) {
    parsing::with_buf(move |buf| {
        if let Some(rem) = amt.checked_sub(buf.len()).and_then(NonZeroUsize::new) {
            amt = rem.get();
            Poll::Pending
        } else {
            Poll::Ready((amt, ()))
        }
    })
    .await
}

fn skip_line_to(buf: &[u8], delim: u8) -> ParseResult<bool> {
    match memchr::memchr3(b'\r', b'\n', delim, buf) {
        Some(index) => {
            let found = buf[index] == delim;
            Poll::Ready((index + found as usize, found))
        }
        None => Poll::Pending,
    }
}

async fn skip_line_to_comma() -> bool {
    parsing::with_buf(|buf| skip_line_to(buf, b',')).await
}

async fn skip_line_to_space() -> bool {
    parsing::with_buf(|buf| skip_line_to(buf, b' ')).await
}

async fn skip_line() -> Result<(), FramingError> {
    parsing::with_buf(|buf| match memchr::memchr2(b'\r', b'\n', buf) {
        Some(index) => Poll::Ready((index, ())),
        None => Poll::Pending,
    })
    .await;

    required_newline().await
}

async fn skip_spaces() {
    parsing::with_buf(|buf| match buf.iter().position(|&b| b != b' ') {
        Some(index) => Poll::Ready((index, ())),
        None => Poll::Pending,
    })
    .await
}

fn expect(buf: &[u8], expected: u8) -> ParseResult<bool> {
    let matched = buf[0] == expected;
    Poll::Ready((matched as usize, matched))
}

async fn colon() -> bool {
    parsing::with_buf(|buf| expect(buf, b':')).await
}

async fn comma() -> bool {
    parsing::with_buf(|buf| expect(buf, b',')).await
}

async fn comma_separated<P, G, F, V>(parser: P, mut f: F) -> Result<(), FramingError>
where
    P: Fn() -> G,
    G: Future<Output = Option<V>>,
    F: FnMut(Option<V>) -> Result<(), FramingError>,
{
    loop {
        let v = parser().await;
        skip_spaces().await;
        if comma().await {
            f(v)?;
        } else if newline().await? {
            f(v)?;
            break;
        } else {
            f(None)?;
            if !skip_line_to_comma().await {
                required_newline().await?;
                break;
            }
        }
        skip_spaces().await;
    }
    Ok(())
}

async fn request_line<M, MV, T>(methods: M, mut target: T) -> Result<(Option<MV>, u8), FramingError>
where
    M: Future<Output = Option<MV>>,
    T: FnMut(&[u8]),
{
    let method = methods.await;
    if method.is_none() {
        if !skip_line_to_space().await {
            return Err(FramingError::BadSyntax);
        }
    }

    parsing::with_buf(move |buf| {
        if let Some(index) = memchr::memchr3(b'\r', b'\n', b' ', buf) {
            if buf[index] != b' ' {
                Poll::Ready((index, Err(FramingError::BadSyntax)))
            } else {
                target(&buf[..index]);
                Poll::Ready((index + 1, Ok(())))
            }
        } else {
            target(buf);
            Poll::Pending
        }
    })
    .await;

    let versions = matcher::matcher(Natural::ORDER, &[(b"HTTP/1.0", 0), (b"HTTP/1.1", 1)]);
    let version = parsing::with_buf(versions)
        .await
        .ok_or(FramingError::BadVersion)?;

    required_newline().await?;

    Ok((method, version))
}

async fn cr() -> bool {
    parsing::with_buf(|buf| expect(buf, b'\r')).await
}

async fn lf() -> bool {
    parsing::with_buf(|buf| expect(buf, b'\n')).await
}

/// Robust check for end-of-line, per
/// <https://tools.ietf.org/html/rfc7230#section-3.5>.
async fn newline() -> Result<bool, FramingError> {
    let saw_cr = cr().await;
    let saw_lf = lf().await;
    if saw_cr && !saw_lf {
        Err(FramingError::BadSyntax)
    } else {
        Ok(saw_lf)
    }
}

async fn required_newline() -> Result<(), FramingError> {
    cr().await;
    if lf().await {
        Ok(())
    } else {
        Err(FramingError::BadSyntax)
    }
}

async fn number(radix: u8) -> Option<usize> {
    let mut value: usize = 0;
    let mut valid = false;
    parsing::with_buf(move |buf| {
        for (idx, digit) in buf.iter().enumerate() {
            if let Some(digit) = char::from(*digit).to_digit(radix.into()) {
                if let Some(last) = value.checked_mul(radix.into()) {
                    // Since to_digit only supports radix 36 or less, `digit` definitely fits in a
                    // u8. That in turn should fit in a usize; make it a compile-time error if not.
                    value = last + usize::from(digit as u8);
                    valid = true;
                } else {
                    return Poll::Ready((idx, None));
                }
            } else {
                let result = if valid { Some(value) } else { None };
                return Poll::Ready((idx, result));
            }
        }
        Poll::Pending
    })
    .await
}

async fn http_date() -> Result<Option<i64>, FramingError> {
    // TODO: parse dates
    skip_line().await?;
    Ok(None)
}

struct ByteRanges;

async fn byte_ranges(ranges: &mut ByteRanges) -> Result<(), FramingError> {
    // TODO: parse byte ranges
    skip_line().await
}

async fn parse_if_range<E, V>(etags: E) -> Result<RangeCondition<V>, FramingError>
where
    E: Future<Output = Option<V>>,
{
    async fn peek_dquote() -> bool {
        parsing::with_buf(|buf| Poll::Ready((0, buf[0] == b'"'))).await
    }

    // Peek one byte ahead to distinguish between etag and date. The RFC says "A valid entity-tag
    // can be distinguished from a valid HTTP-date by examining the first two characters for a
    // DQUOTE," but since it's required to be a strong validator, I can't see why the first
    // (non-space) character isn't enough.
    if peek_dquote().await {
        if let Some(etag) = etags.await {
            skip_spaces().await;
            required_newline().await?;
            Ok(RangeCondition::ETag(etag))
        } else {
            skip_line().await?;
            Ok(RangeCondition::Failed)
        }
    } else {
        if let Some(date) = http_date().await? {
            Ok(RangeCondition::LastModified(date))
        } else {
            Ok(RangeCondition::Failed)
        }
    }
}

struct MessageHeader<'a, E> {
    etags: Vec<(&'a [u8], E)>,
    body: Body,
    persistent: bool,
    if_none_match: HashSet<E>,
    if_modified_since: Option<i64>,
    range: ByteRanges,
    if_range: Option<RangeCondition<E>>,
}

impl<'a, E: Copy + Eq + std::hash::Hash> MessageHeader<'a, E> {
    pub fn new(etags: Vec<(&'a [u8], E)>, version: u8) -> Self {
        MessageHeader {
            etags,
            body: Body::None,
            persistent: version >= 1,
            if_none_match: HashSet::new(),
            if_modified_since: None,
            range: ByteRanges,
            if_range: None,
        }
    }

    pub async fn parse_header_value(&mut self, header: Header) -> Result<(), FramingError> {
        match header {
            // https://tools.ietf.org/html/rfc7230#section-3.3.2
            Header::ContentLength => {
                comma_separated(
                    || number(10),
                    |length| {
                        if let Some(length) = length {
                            self.body.update(Body::Length(length))
                        } else {
                            Err(FramingError::BadSyntax)
                        }
                    },
                )
                .await
            }

            // https://tools.ietf.org/html/rfc7230#section-3.3.1
            Header::TransferEncoding => {
                // If the request is using Transfer-Encoding, the "chunked" encoding
                // must be present, exactly once, at the end of the list. We don't care
                // about any others because we don't interpret the message body.
                let transfer_codings = matcher::matcher(CaseInsensitiveASCII, &[(b"chunked", ())]);
                comma_separated(
                    || parsing::with_buf(transfer_codings.clone()),
                    |value| {
                        self.body.update(if value.is_some() {
                            Body::Chunked
                        } else {
                            Body::None
                        })
                    },
                )
                .await?;

                if !matches!(self.body, Body::Chunked) {
                    Err(FramingError::BadSyntax)
                } else {
                    Ok(())
                }
            }

            // https://tools.ietf.org/html/rfc7230#section-6.1
            Header::Connection => {
                let keepalive_options = matcher::matcher(
                    CaseInsensitiveASCII,
                    &[(b"close", false), (b"keep-alive", true)],
                );
                comma_separated(
                    || parsing::with_buf(keepalive_options.clone()),
                    |value| {
                        if let Some(keepalive) = value {
                            self.persistent = keepalive;
                        }
                        Ok(())
                    },
                )
                .await
            }

            // https://tools.ietf.org/html/rfc7232#section-3.2
            Header::IfNoneMatch => {
                let mut if_none_match = std::mem::replace(&mut self.if_none_match, HashSet::new());
                let etags = matcher::matcher(Natural::ORDER, &self.etags);
                comma_separated(
                    || parsing::with_buf(etags.clone()),
                    |value| {
                        // We only care about ETags that match some representation we currently
                        // have for this resource.
                        if let Some(value) = value {
                            if_none_match.insert(value);
                        }
                        Ok(())
                    },
                )
                .await?;
                self.if_none_match = if_none_match;
                Ok(())
            }

            // https://tools.ietf.org/html/rfc7232#section-3.3
            Header::IfModifiedSince => {
                self.if_modified_since = http_date().await?;
                Ok(())
            }

            // https://tools.ietf.org/html/rfc7233#section-3.1
            Header::Range => byte_ranges(&mut self.range).await,

            // https://tools.ietf.org/html/rfc7233#section-3.2
            Header::IfRange => {
                let etags = matcher::matcher(Natural::ORDER, &self.etags);
                self.if_range = Some(parse_if_range(parsing::with_buf(etags.clone())).await?);
                Ok(())
            }
        }
    }

    pub async fn finish(&self) -> Result<(), FramingError> {
        match self.body {
            Body::None => {}
            Body::Length(n) => skip(n).await,
            Body::Chunked => {
                // https://tools.ietf.org/html/rfc7230#section-4.1
                loop {
                    let len = number(16).await.ok_or(FramingError::BadSyntax)?;
                    skip_line().await?;
                    if len == 0 {
                        break;
                    }
                    skip(len).await;
                    required_newline().await?;
                }

                while !newline().await? {
                    skip_line().await?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Method {
    Get,
    Head,
}

#[derive(Clone, Copy)]
enum Header {
    ContentLength,
    Connection,
    IfModifiedSince,
    IfNoneMatch,
    IfRange,
    Range,
    TransferEncoding,
}

static BASE_HEADERS: [(&[u8], Header); 7] = [
    (b"connection:", Header::Connection),
    (b"content-length:", Header::ContentLength),
    (b"if-modified-since:", Header::IfModifiedSince),
    (b"if-none-match:", Header::IfNoneMatch),
    (b"if-range:", Header::IfRange),
    (b"range:", Header::Range),
    (b"transfer-encoding:", Header::TransferEncoding),
];

enum Body {
    None,
    Chunked,
    Length(usize),
}

impl Body {
    // Reject if there's more than one Transfer-Encoding header, or there's more than one distinct
    // Content-Length, or both headers are present.
    // https://tools.ietf.org/html/rfc7230#section-3.3.2
    fn update(&mut self, new: Self) -> Result<(), FramingError> {
        match (&*self, new) {
            (Body::None, new) => {
                *self = new;
                Ok(())
            }
            (Body::Length(old), Body::Length(new)) if *old == new => Ok(()),
            _ => Err(FramingError::BadSyntax),
        }
    }
}

enum RangeCondition<E> {
    Failed,
    ETag(E),
    LastModified(i64),
}

async fn message<R, W>(reader: &mut R, writer: &mut W) -> Result<(), FramingError>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    // Support only GET and HEAD methods (and maybe TRACE?). Seems like OPTIONS (e.g. CORS) really
    // only makes sense with a dynamic backend.
    let methods = matcher::matcher(
        Natural::ORDER,
        &[(b"GET ", Method::Get), (b"HEAD ", Method::Head)],
    );

    let mut req_hash = Blake2s::new();
    let (method, version) = fold(
        reader,
        request_line(parsing::with_buf(methods), |buf| req_hash.update(buf)),
    )
    .await?;

    // TODO: open a resource, at `path`, or user-defined 404, or built-in 404
    let path = base64::encode_config(&req_hash.finalize(), base64::URL_SAFE_NO_PAD);
    let resource = ();

    let mut known_headers = Cow::from(&BASE_HEADERS[..]);

    // TODO: if the selected resource has any negotiation headers ...
    let negotiations = Vec::new();
    if !negotiations.is_empty() {
        let known_headers = known_headers.to_mut();
        known_headers.extend_from_slice(&negotiations);
        known_headers.sort_unstable_by(|(a, _), (b, _)| CaseInsensitiveASCII.cmp(a, b));
    }

    // TODO: if the selected resource has any representations with etags ...
    let etags: Vec<(&[u8], ())> = Vec::new();

    let known_headers = matcher::matcher(CaseInsensitiveASCII, &known_headers);

    let message_header = fold(reader, async {
        let mut message_header = MessageHeader::new(etags, version);
        while !newline().await? {
            if let Some(v) = parsing::with_buf(known_headers.clone()).await {
                skip_spaces().await;
                message_header.parse_header_value(v).await?;
                continue;
            }
            skip_line().await?;
        }
        Ok(message_header)
    })
    .await?;

    if let Err(_) = tokio::try_join!(
        fold(reader, message_header.finish()),
        send_response(writer, ()),
    ) {
        Err(FramingError::End)
    } else if message_header.persistent {
        Ok(())
    } else {
        Err(FramingError::End)
    }
}

const ERROR_500: &[u8] =
    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 9\r\n\r\nNot yet!\n";

async fn send_response<W>(writer: &mut W, response: ()) -> Result<(), FramingError>
where
    W: AsyncWrite + Unpin,
{
    if let Err(_) = writer.write_all(ERROR_500).await {
        Err(FramingError::End)
    } else {
        Ok(())
    }
}

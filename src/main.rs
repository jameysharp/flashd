use futures::TryFutureExt;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::future::Future;
use std::io::Write;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::task::Poll;
use tokio::fs::File;
use tokio::io::{
    copy, AsyncBufRead, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::net::TcpListener;

mod matcher;
use matcher::{CaseInsensitiveASCII, Comparator, Natural};

mod parsing;

#[macro_use]
mod framing;
use framing::{fold, FramingError, ParseResult};

#[allow(non_snake_case, unused)]
mod resource {
    include!(concat!(env!("OUT_DIR"), "/resource_generated.rs"));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (socket, _) = listener.accept().await?;
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
            // no representation-specific headers, close connection after sending:
            // - 400 Bad Request
            // - 408 Request Timeout
            // - 505 HTTP Version Not Supported
            match e {
                FramingError::End => {}
                FramingError::BadSyntax => {
                    let _ = Response {
                        head: false,
                        close: true,
                        status: 400,
                        vary: Vec::new(),
                        header_length: 2,
                        source: ResponseSource::Buffer(b"\r\nBad syntax in HTTP request\n"),
                    }
                    .send(&mut writer)
                    .await;
                }
                FramingError::BadVersion => {
                    let _ = Response {
                        head: false,
                        close: true,
                        status: 505,
                        vary: Vec::new(),
                        header_length: 2,
                        source: ResponseSource::Buffer(b"\r\nHTTP version not supported\n"),
                    }
                    .send(&mut writer)
                    .await;
                }
            }
            break;
        }
    }

    let _ = writer.shutdown().await;
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
    .await?;

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

async fn byte_ranges(_ranges: &mut ByteRanges) -> Result<(), FramingError> {
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

/// Fixed-point representation of qvalues, which are defined to range from 0.0 to 1.0 with three
/// decimal digits of precision.
///
/// See [RFC7231 section 5.3.1](https://tools.ietf.org/html/rfc7231#section-5.3.1).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct QValue(u16);

impl QValue {
    const ONE: Self = QValue(1000);
    const ZERO: Self = QValue(0);
}

impl std::ops::Mul for QValue {
    type Output = QValue;
    fn mul(self, rhs: Self) -> Self {
        // Widen to at least 20 bits so the intermediate product can't overflow:
        let lhs: u32 = self.0.into();
        let rhs: u32 = rhs.0.into();
        // Take ceil(lhs*rhs) so multiplying non-zero qvalues will never produce 0:
        let ceil = (lhs * rhs + 999) / 1000;
        // After dividing by 1000, the intermediate product is back in the correct range:
        QValue(ceil as u16)
    }
}

enum NegotiationWildcard {
    NotNegotiated,
    NoWildcard,
    QValue(QValue),
}

struct NegotiationState<'a> {
    negotiation: resource::Negotiation<'a>,
    qvalues: HashMap<u16, (u8, QValue)>,
    wildcard: NegotiationWildcard,
}

struct MessageHeader<'a> {
    etags: Vec<(&'a [u8], usize)>,
    body: Body,
    persistent: bool,
    if_none_match: HashSet<usize>,
    if_modified_since: Option<i64>,
    range: ByteRanges,
    if_range: Option<RangeCondition<usize>>,
    negotiations: Vec<NegotiationState<'a>>,
}

impl<'a> MessageHeader<'a> {
    pub fn new(version: u8, resource: resource::Resource<'a>) -> Self {
        let mut etags = resource
            .representations()
            .into_iter()
            .enumerate()
            .filter_map(|(idx, representation)| {
                representation.etag().map(|etag| (etag.as_bytes(), idx))
            })
            .collect::<Vec<_>>();
        etags.sort_unstable();

        let negotiations = if let Some(negotiations) = resource.negotiations() {
            negotiations
                .into_iter()
                .map(|negotiation| NegotiationState {
                    negotiation,
                    qvalues: HashMap::new(),
                    wildcard: NegotiationWildcard::NotNegotiated,
                })
                .collect()
        } else {
            Vec::new()
        };

        MessageHeader {
            etags,
            body: Body::None,
            persistent: version >= 1,
            if_none_match: HashSet::new(),
            if_modified_since: None,
            range: ByteRanges,
            if_range: None,
            negotiations,
        }
    }

    fn get_headers(&self) -> Cow<'a, [(&'a [u8], Header)]> {
        let mut known_headers = Cow::from(&BASE_HEADERS[..]);
        if !self.negotiations.is_empty() {
            let known_headers = known_headers.to_mut();
            known_headers.extend(
                self.negotiations
                    .iter()
                    .enumerate()
                    .map(|(idx, n)| (n.negotiation.header().as_bytes(), Header::Negotiation(idx))),
            );
            known_headers.sort_unstable_by(|(a, _), (b, _)| CaseInsensitiveASCII.cmp(a, b));
        }
        known_headers
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
                let MessageHeader {
                    if_none_match,
                    etags,
                    ..
                } = self;
                let etags = matcher::matcher(Natural::ORDER, etags);
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
                .await
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

            Header::Negotiation(idx) => {
                let NegotiationState {
                    negotiation,
                    qvalues,
                    wildcard,
                } = &mut self.negotiations[idx];

                if matches!(wildcard, NegotiationWildcard::NotNegotiated) {
                    // This is the first copy of this header we've seen.
                    *wildcard = NegotiationWildcard::NoWildcard;
                }

                let mut values = vec![(
                    negotiation.wildcard().map_or(&b"*"[..], str::as_bytes),
                    None,
                )];
                values.extend(
                    negotiation
                        .choices()
                        .into_iter()
                        .map(|choice| (choice.name().as_bytes(), Some(choice))),
                );
                values.sort_unstable_by(|(a, _), (b, _)| CaseInsensitiveASCII.cmp(a, b));

                let values = matcher::matcher(CaseInsensitiveASCII, &values);

                comma_separated(
                    // TODO: parse qvalue, skip extensions
                    || parsing::with_buf(values.clone()),
                    |value| {
                        // TODO: insert parsed qvalue
                        let qvalue = QValue::ONE;
                        match value {
                            Some(Some(choice)) => {
                                let new = (choice.specificity(), qvalue);
                                for representation in choice.representations() {
                                    let old =
                                        qvalues.entry(representation).or_insert((0, QValue::ZERO));
                                    if new > *old {
                                        *old = new;
                                    }
                                }
                            }
                            Some(None) => {
                                *wildcard = NegotiationWildcard::QValue(qvalue);
                            }
                            None => {}
                        }
                        Ok(())
                    },
                )
                .await
            }
        }
    }

    fn select_best_representation(&self) -> Option<usize> {
        let mut qualified = HashMap::new();
        let mut wildcard = QValue::ONE;

        for state in self.negotiations.iter() {
            // First, check if there was a negotiation header we can't satisfy.
            if state.qvalues.is_empty() {
                match state.wildcard {
                    // Client specified that anything they didn't list is unacceptable:
                    NegotiationWildcard::QValue(QValue::ZERO) => return None,

                    // Client didn't specify wildcard, and server says it must match:
                    NegotiationWildcard::NoWildcard if state.negotiation.must_match() => return None,

                    // Applying the same non-zero qvalue to all representations,
                    NegotiationWildcard::QValue(_) |
                    // or not specifying a wildcard when the server says matching is optional,
                    NegotiationWildcard::NoWildcard |
                    // are equivalent to not specifying the header at all:
                    NegotiationWildcard::NotNegotiated => {}
                }
                continue;
            }

            let default = match state.wildcard {
                // Something matched, so the header must have been present.
                NegotiationWildcard::NotNegotiated => unreachable!(),

                // Something matched, so we should only select one of the matches:
                NegotiationWildcard::NoWildcard => QValue::ZERO,

                // Unless the client said otherwise:
                NegotiationWildcard::QValue(v) => v,
            };

            for (idx, qvalue) in qualified.iter_mut() {
                let new = state
                    .qvalues
                    .get(idx)
                    .map_or(default, |(_, qvalue)| *qvalue);
                *qvalue = *qvalue * new;
            }

            for (idx, (_, qvalue)) in state.qvalues.iter() {
                if *qvalue != default {
                    qualified.entry(*idx).or_insert(wildcard * *qvalue);
                }
            }

            wildcard = wildcard * default;
        }

        // Pick the representation with the highest qvalue; break ties with the lowest index. But
        // only pick representations that beat the composite wildcard qvalue.
        let best = qualified
            .iter()
            .filter(|(_, qvalue)| **qvalue > wildcard)
            .max_by(|(idx_a, qvalue_a), (idx_b, qvalue_b)| {
                qvalue_a.cmp(qvalue_b).then(idx_b.cmp(idx_a))
            });

        if let Some((best_idx, qvalue)) = best {
            // This qvalue is strictly greater than wildcard, and wildcard is no smaller than zero.
            debug_assert_ne!(*qvalue, QValue::ZERO);
            Some((*best_idx).into())
        } else if wildcard > QValue::ZERO {
            // If nothing is better than the wildcard, then we need to pick the first
            // representation that equals the wildcard, which might not appear in the `qualified`
            // map.
            let mut best_idx = 0;
            loop {
                let good = qualified.get(&best_idx).map_or(true, |qvalue| *qvalue == wildcard);
                if good {
                    break Some(best_idx.into());
                }
                if let Some(next) = best_idx.checked_add(1) {
                    best_idx = next;
                } else {
                    break None;
                }
            }
        } else {
            // The best option we could find had a qvalue of 0, which means the client does not
            // consider it acceptable and we shouldn't return anything.
            None
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
    Negotiation(usize),
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

    let mut req_hash = Sha256::new();
    let (method, version) = fold(
        reader,
        request_line(parsing::with_buf(methods), |buf| req_hash.update(buf)),
    )
    .await?;

    let path = base64::encode_config(&req_hash.finalize(), base64::URL_SAFE_NO_PAD);

    // open a resource, at `path`, or user-defined 404, or built-in 404
    let resource_buf = if let Ok(buf) = tokio::fs::read(path).await {
        Cow::from(buf)
    } else if let Ok(buf) = tokio::fs::read("error404").await {
        Cow::from(buf)
    } else {
        Cow::from(&include_bytes!(concat!(env!("OUT_DIR"), "/404.http"))[..])
    };
    let resource = resource::get_root_as_resource(&resource_buf);

    let message_header = fold(reader, async {
        let mut message_header = MessageHeader::new(version, resource);
        let known_headers = message_header.get_headers();
        let known_headers = matcher::matcher(CaseInsensitiveASCII, &known_headers);
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

    // TODO: use message_header to decide how to respond.
    // send headers from matching representation:
    // - 206 Partial Content
    // - 304 Not Modified
    // no representation-specific headers:
    // - 405 Method Not Allowed (include "Allow: GET, HEAD")
    // - 406 Not Acceptable

    let close = !message_header.persistent;
    let response = if let Some(method) = method {
        let head = matches!(method, Method::Head);

        let vary = if let Some(negotiations) = resource.negotiations() {
            negotiations.iter().map(|negotiation| negotiation.header().as_bytes()).collect()
        } else {
            Vec::new()
        };

        match message_header.select_best_representation() {
            Some(idx) if idx < resource.representations().len() => {
                let representation = resource.representations().get(idx);

                let not_modified = message_header.if_none_match.contains(&idx);

                let source = if let Some(source) = representation.source_as_file_source() {
                    if let Ok(file) = tokio::fs::File::open(source.filename()).await {
                        Some(ResponseSource::File(file))
                    } else {
                        None
                    }
                } else if let Some(source) = representation.source_as_inline_source() {
                    Some(ResponseSource::Buffer(source.contents().as_bytes()))
                } else {
                    None
                };

                if let Some(source) = source {
                    Response {
                        head: head || not_modified,
                        close,
                        status: if not_modified {
                            304
                        } else {
                            representation.status()
                        },
                        vary,
                        header_length: representation.header_length(),
                        source,
                    }
                } else {
                    Response {
                        head,
                        close,
                        status: 500,
                        vary,
                        header_length: 2,
                        source: ResponseSource::Buffer(b"\r\nCouldn't load selected representation\n"),
                    }
                }
            }

            _ => {
                Response {
                    head,
                    close,
                    status: 406,
                    vary,
                    header_length: 2,
                    source: ResponseSource::Buffer(b"\r\nThis resource has no representation that can satisfy your request.\n"),
                }
            }
        }
    } else {
        Response {
            head: false,
            close,
            status: 405,
            vary: Vec::new(),
            header_length: 20,
            source: ResponseSource::Buffer(b"Allow: GET, HEAD\r\n\r\nRequest method not allowed\n"),
        }
    };

    if let Err(_) = tokio::try_join!(
        fold(reader, message_header.finish()),
        response.send(writer).map_err(|_| FramingError::End),
    ) {
        Err(FramingError::End)
    } else if message_header.persistent {
        Ok(())
    } else {
        Err(FramingError::End)
    }
}

#[allow(dead_code)]
enum ResponseSource<'a> {
    Buffer(&'a [u8]),
    File(File),
}

impl ResponseSource<'_> {
    async fn len(&self) -> std::io::Result<u64> {
        Ok(match self {
            ResponseSource::Buffer(buf) => buf.len().try_into().unwrap(),
            ResponseSource::File(f) => f.metadata().await?.len(),
        })
    }
}

impl AsyncRead for ResponseSource<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context,
        buf: &mut tokio::io::ReadBuf,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            ResponseSource::Buffer(b) => Pin::new(b).poll_read(cx, buf),
            ResponseSource::File(f) => Pin::new(f).poll_read(cx, buf),
        }
    }
}

struct Response<'a> {
    head: bool,
    close: bool,
    status: u16,
    vary: Vec<&'a [u8]>,
    header_length: u64,
    source: ResponseSource<'a>,
}

impl Response<'_> {
    pub async fn send<W>(self, writer: &mut W) -> std::io::Result<()>
    where
        W: AsyncWrite + Unpin,
    {
        // add response headers:
        // - Connection: close (depends on whether we understood the request)
        // - Content-Length (varies depending on byte-ranges)
        // - Date (necessarily dynamic)
        // - Vary (a property of the resource, not the representation)

        let source_length = self.source.len().await?;
        let mut buffer = Vec::new();
        write!(
            &mut buffer,
            "HTTP/1.1 {} \r\nContent-Length: {}\r\n",
            self.status,
            source_length - self.header_length,
        )?;

        let mut vary = self.vary.into_iter();
        if let Some(header) = vary.next() {
            buffer.extend_from_slice(b"Vary: ");
            buffer.extend_from_slice(&header[..header.len() - 1]);
            for header in vary {
                buffer.extend_from_slice(b", ");
                buffer.extend_from_slice(&header[..header.len() - 1]);
            }
            buffer.extend_from_slice(b"\r\n");
        }

        if self.close {
            buffer.extend_from_slice(b"Connection: close\r\n");
        }

        // It should be unnecessary to call `take(source_length)` since that's exactly the length
        // we believe we'll get if we read everything. But type-checking is simpler if the `Take`
        // adaptor is part of the pipeline in all cases; this prevents us from sending garbage to
        // the client if the file gets longer while we're reading it; and compared to the cost of
        // I/O, the arithmetic involved is basically free.
        let source = self.source.take(if self.head {
            self.header_length
        } else {
            source_length
        });

        copy(&mut (&buffer[..]).chain(source), writer).await?;
        Ok(())
    }
}

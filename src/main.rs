#![allow(unused)]

use blake2::{Blake2s, Digest};
use std::borrow::Cow;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::task::Poll;
use tokio::io::AsyncBufRead;
use tokio::net::TcpListener;

mod matcher;
use matcher::{CaseInsensitiveASCII, Comparator, Natural};

#[macro_use]
mod framing;
use framing::{fold, FramingError, ParseResult};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;
    }
}

fn skip(mut amt: usize) -> impl FnMut(&[u8]) -> ParseResult<()> {
    move |buf| {
        if let Some(rem) = amt.checked_sub(buf.len()).and_then(NonZeroUsize::new) {
            amt = rem.get();
            Poll::Pending
        } else {
            Poll::Ready((amt, ()))
        }
    }
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

fn skip_line_to_comma(buf: &[u8]) -> ParseResult<bool> {
    skip_line_to(buf, b',')
}

fn skip_line_to_space(buf: &[u8]) -> ParseResult<bool> {
    skip_line_to(buf, b' ')
}

fn skip_line() -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>> {
    fn skip_until_newline(buf: &[u8]) -> ParseResult<()> {
        match memchr::memchr2(b'\r', b'\n', buf) {
            Some(index) => Poll::Ready((index, ())),
            None => Poll::Pending,
        }
    }

    stateful! {
        Skip() => impl<N> {
            for Skip() match skip_until_newline => () { () => Newline() }
            for Newline() let p: N = required_newline(); match p => Result<(), FramingError> { v => break v }
        }
    }
}

fn skip_spaces(buf: &[u8]) -> ParseResult<()> {
    match buf.iter().position(|&b| b != b' ') {
        Some(index) => Poll::Ready((index, ())),
        None => Poll::Pending,
    }
}

fn expect(buf: &[u8], expected: u8) -> ParseResult<bool> {
    let matched = buf[0] == expected;
    Poll::Ready((matched as usize, matched))
}

fn colon(buf: &[u8]) -> ParseResult<bool> {
    expect(buf, b':')
}

fn comma(buf: &[u8]) -> ParseResult<bool> {
    expect(buf, b',')
}

fn comma_separated<P, F, V>(parser: P, mut f: F) -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>>
where
    P: FnMut(&[u8]) -> ParseResult<Option<V>> + Clone,
    F: FnMut(Option<V>) -> Result<(), FramingError>,
{
    stateful! {
        Parse() => impl<V, P, N, E> {
            for Parse() let p: P = parser.clone(); match p => Option<V> {
                v => SpaceBefore(v)
            }
            for SpaceBefore(v: Option<V>) match skip_spaces => () {
                () => Comma(v)
            }
            for Comma(v: Option<V>) match comma => bool {
                true => {
                    if let Err(e) = f(v) { break Err(e); }
                    SpaceAfter()
                },
                false => CheckEnd(v),
            }
            for CheckEnd(v: Option<V>) let p: N = newline(); match p => Result<bool, FramingError> {
                Err(e) => break Err(e),
                Ok(true) => {
                    if let Err(e) = f(v) { break Err(e); }
                    break Ok(());
                },
                Ok(false) => {
                    if let Err(e) = f(None) { break Err(e); }
                    Skip()
                },
            }
            for Skip() match skip_line_to_comma => bool {
                true => SpaceAfter(),
                false => End(),
            }
            for SpaceAfter() match skip_spaces => () {
                () => Parse()
            }
            for End() let p: E = required_newline(); match p => Result<(), FramingError> {
                v => break v
            }
        }
    }
}

fn request_target<F>(mut consumer: F) -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>>
where
    F: FnMut(&[u8]),
{
    move |buf| {
        if let Some(index) = memchr::memchr3(b'\r', b'\n', b' ', buf) {
            if buf[index] != b' ' {
                Poll::Ready((index, Err(FramingError::BadSyntax)))
            } else {
                consumer(&buf[..index]);
                Poll::Ready((index + 1, Ok(())))
            }
        } else {
            consumer(buf);
            Poll::Pending
        }
    }
}

fn request_line<M, MV, T, TV>(
    methods: M,
    target: T,
) -> impl FnMut(&[u8]) -> ParseResult<Result<(Option<MV>, TV, u8), FramingError>>
where
    M: FnMut(&[u8]) -> ParseResult<Option<MV>>,
    T: FnMut(&[u8]) -> ParseResult<Result<TV, FramingError>>,
{
    stateful! {
        Method(methods, target) => impl<M, MV, T, TV, V, N> {
            for Method(methods: M, target: T) match methods => Option<MV> {
                None => SkipMethod(target),
                Some(method) => Target(Some(method), target),
            }
            for SkipMethod(target: T) match skip_line_to_space => bool {
                false => break Err(FramingError::BadSyntax),
                true => Target(None, target),
            }
            for Target(method: Option<MV>, target: T)
            match target => Result<TV, FramingError> {
                Err(e) => break Err(e),
                Ok(target) => Version(method, target)
            }
            for Version(method: Option<MV>, target: TV)
            let versions: V = matcher::matcher(Natural::ORDER, &[(b"HTTP/1.0", 0), (b"HTTP/1.1", 1)]);
            match versions => Option<u8> {
                Some(version) => Newline(method, target, version),
                None => break Err(FramingError::BadVersion),
            }
            for Newline(method: Option<MV>, target: TV, version: u8)
            let p: N = required_newline();
            match p => Result<(), FramingError> {
                Err(e) => break Err(e),
                Ok(()) => break Ok((method, target, version)),
            }
        }
    }
}

fn headers<P, F, N, V>(name: P, mut f: F) -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>>
where
    P: FnMut(&[u8]) -> ParseResult<Option<N>> + Clone,
    F: FnMut(N) -> V,
    N: Copy,
    V: FnMut(&[u8]) -> ParseResult<()>,
{
    stateful! {
        CheckEnd(name) => impl<L, N, P, V, S> {
            for CheckEnd(name: P) let p: L = newline(); match p => Result<bool, FramingError> {
                Err(e) => break Err(e),
                Ok(true) => break Ok(()),
                Ok(false) => Name(name),
            }
            for Name(name: P) let p: P = name.clone(); match p => Option<N> {
                None => Skip(name),
                Some(v) => Colon(name, v),
            }
            for Colon(name: P, v: N) match colon => bool {
                false => Skip(name),
                true => Spaces(name, v),
            }
            for Spaces(name: P, v: N) match skip_spaces => () {
                () => Value(name, f(v))
            }
            for Value(name: P, p: V) match p => () {
                () => CheckEnd(name)
            }
            for Skip(name: P) let p: S = skip_line(); match p => Result<(), FramingError> {
                Ok(()) => CheckEnd(name),
                Err(e) => break Err(e),
            }
        }
    }
}

fn cr(buf: &[u8]) -> ParseResult<bool> {
    expect(buf, b'\r')
}

fn lf(buf: &[u8]) -> ParseResult<bool> {
    expect(buf, b'\n')
}

/// Robust check for end-of-line, per
/// <https://tools.ietf.org/html/rfc7230#section-3.5>.
fn newline() -> impl FnMut(&[u8]) -> ParseResult<Result<bool, FramingError>> {
    stateful! {
        Start() => impl {
            for Start() match cr => bool {
                true => RequireLF(),
                false => OptionalLF(),
            }
            for RequireLF() match lf => bool {
                true => break Ok(true),
                false => break Err(FramingError::BadSyntax),
            }
            for OptionalLF() match lf => bool {
                found => break Ok(found),
            }
        }
    }
}

fn required_newline() -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>> {
    stateful! {
        Start() => impl {
            for Start() match cr => bool { _ => RequireLF() }
            for RequireLF() match lf => bool {
                true => break Ok(()),
                false => break Err(FramingError::BadSyntax),
            }
        }
    }
}

fn number(radix: u8) -> impl FnMut(&[u8]) -> ParseResult<Option<usize>> + Clone {
    let mut value: usize = 0;
    let mut valid = false;
    move |buf| {
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
    }
}

fn http_date() -> impl FnMut(&[u8]) -> ParseResult<Result<Option<i64>, FramingError>> {
    // TODO: parse dates
    stateful! {
        Skip() => impl<N> {
            for Skip() let p: N = skip_line(); match p => Result<(), FramingError> {
                Ok(()) => break Ok(None),
                Err(e) => break Err(e),
            }
        }
    }
}

struct ByteRanges;

fn byte_ranges(ranges: &mut ByteRanges) -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>> {
    // TODO: parse byte ranges
    skip_line()
}

fn parse_if_range<E>(etags: E) -> impl FnMut(&[u8]) -> ParseResult<Result<RangeCondition, FramingError>>
where
    E: FnMut(&[u8]) -> ParseResult<Option<()>>,
{
    fn peek_dquote(buf: &[u8]) -> ParseResult<bool> {
        Poll::Ready((0, buf[0] == b'"'))
    }

    stateful! {
        CheckQuote(etags) => impl<D, E, N, S> {
            // Peek one byte ahead to distinguish between etag and date. The RFC says "A valid
            // entity-tag can be distinguished from a valid HTTP-date by examining the first two
            // characters for a DQUOTE," but since it's required to be a strong validator, I can't
            // see why the first (non-space) character isn't enough.
            for CheckQuote(etags: E) match peek_dquote => bool {
                true => Etag(etags),
                false => Date(),
            }
            for Etag(etags: E) match etags => Option<()> {
                Some(etag) => Spaces(etag),
                None => Skip(),
            }
            for Spaces(etag: ()) match skip_spaces => () {
                () => Newline(etag)
            }
            for Newline(etag: ()) let p: N = required_newline(); match p => Result<(), FramingError> {
                Err(e) => break Err(e),
                Ok(()) => break Ok(RangeCondition::ETag(etag)),
            }
            for Skip() let p: S = skip_line(); match p => Result<(), FramingError> {
                Err(e) => break Err(e),
                Ok(()) => break Ok(RangeCondition::Failed)
            }
            for Date() let date: D = http_date(); match date => Result<Option<i64>, FramingError> {
                Err(e) => break Err(e),
                Ok(Some(date)) => break Ok(RangeCondition::LastModified(date)),
                Ok(None) => break Ok(RangeCondition::Failed),
            }
        }
    }
}

// https://tools.ietf.org/html/rfc7230#section-4.1
fn chunked() -> impl FnMut(&[u8]) -> ParseResult<Result<(), FramingError>> {
    stateful! {
        Chunk() => impl<N, L, S, E, T> {
            for Chunk() let p: N = number(16); match p => Option<usize> {
                None => break Err(FramingError::BadSyntax),
                Some(len) => Line(len),
            }
            for Line(len: usize) let p: L = skip_line(); match p => Result<(), FramingError> {
                Err(e) => break Err(e),
                Ok(()) => if len > 0 {
                    Skip(len)
                } else {
                    Trailer()
                }
            }
            for Skip(len: usize) let p: S = skip(len); match p => () { () => EndChunk() }
            for EndChunk() let p: E = required_newline(); match p => Result<(), FramingError> {
                Err(e) => break Err(e),
                Ok(()) => Chunk(),
            }
            for Trailer() let p: T = newline(); match p => Result<bool, FramingError> {
                Err(e) => break Err(e),
                Ok(true) => break Ok(()),
                Ok(false) => SkipTrailer(),
            }
            for SkipTrailer() let p: L = skip_line(); match p => Result<(), FramingError> {
                Ok(()) => Trailer(),
                Err(e) => break Err(e),
            }
        }
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
    (b"content-length", Header::ContentLength),
    (b"connection", Header::Connection),
    (b"if-modified-since", Header::IfModifiedSince),
    (b"if-none-match", Header::IfNoneMatch),
    (b"if-range", Header::IfRange),
    (b"range", Header::Range),
    (b"transfer-encoding", Header::TransferEncoding),
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

enum RangeCondition {
    Failed,
    ETag(()),
    LastModified(i64),
}

fn ok<F, V>(mut f: F) -> impl FnMut(&[u8]) -> ParseResult<Result<V, FramingError>>
where
    F: FnMut(&[u8]) -> ParseResult<V>,
{
    move |buf| match f(buf) {
        Poll::Ready((consumed, v)) => Poll::Ready((consumed, Ok(v))),
        Poll::Pending => Poll::Pending,
    }
}

async fn message<T: AsyncBufRead + Unpin>(reader: &mut T) -> Result<bool, FramingError> {
    // Support only GET and HEAD methods (and maybe TRACE?). Seems like OPTIONS (e.g. CORS) really
    // only makes sense with a dynamic backend.
    let methods = matcher::matcher(
        Natural::ORDER,
        &[(b"GET ", Method::Get), (b"HEAD ", Method::Head)],
    );

    let mut req_hash = Blake2s::new();
    let (method, (), version) = fold(
        reader,
        &mut request_line(methods, request_target(|buf| req_hash.update(buf))),
    )
    .await?;

    // TODO: open a resource, at `path`, or user-defined 404, or built-in 404
    let path = base64::encode_config(&req_hash.finalize(), base64::URL_SAFE_NO_PAD);
    let resource = ();

    let mut headers = Cow::from(&BASE_HEADERS[..]);

    // TODO: if the selected resource has any negotiation headers ...
    let negotiations = Vec::new();
    if !negotiations.is_empty() {
        let headers = headers.to_mut();
        headers.extend_from_slice(&negotiations);
        headers.sort_unstable_by(|(a, _), (b, _)| CaseInsensitiveASCII.cmp(a, b));
    }

    // TODO: if the selected resource has any representations with etags ...
    let etags: Vec<(&[u8], ())> = Vec::new();

    let headers = matcher::matcher(CaseInsensitiveASCII, &headers);
    let etags = matcher::matcher(Natural::ORDER, &etags);

    let transfer_codings = matcher::matcher(CaseInsensitiveASCII, &[(b"chunked", ())]);
    let keepalive_options = matcher::matcher(
        CaseInsensitiveASCII,
        &[(b"close", false), (b"keep-alive", true)],
    );

    // Requests have a body iff either Content-Length or Transfer-Encoding is present.
    // https://tools.ietf.org/html/rfc7230#section-3.3
    let mut body = Body::None;

    // Request persistence depends on the HTTP version of the request and the presence of either
    // "close" or "keep-alive" options in a Connection header.
    // https://tools.ietf.org/html/rfc7230#section-6.3
    let mut persistent = version >= 1;

    // https://tools.ietf.org/html/rfc7232#section-3.2
    let mut if_none_match = HashSet::new();

    // https://tools.ietf.org/html/rfc7232#section-3.3
    let mut if_modified_since = None;

    // https://tools.ietf.org/html/rfc7233#section-3.1
    let mut range = ByteRanges;

    // https://tools.ietf.org/html/rfc7233#section-3.2
    let mut if_range = None;

    while !fold(reader, &mut newline()).await? {
        if let Some(header) = fold(reader, &mut ok(headers.clone())).await? {
            if fold(reader, &mut ok(colon)).await? {
                fold(reader, &mut ok(skip_spaces)).await?;
                match header {
                    // https://tools.ietf.org/html/rfc7230#section-3.3.2
                    Header::ContentLength => {
                        fold(
                            reader,
                            &mut comma_separated(number(10), |length| {
                                if let Some(length) = length {
                                    body.update(Body::Length(length))
                                } else {
                                    Err(FramingError::BadSyntax)
                                }
                            }),
                        )
                        .await?;
                    }

                    // https://tools.ietf.org/html/rfc7230#section-3.3.1
                    Header::TransferEncoding => {
                        // If the request is using Transfer-Encoding, the "chunked" encoding
                        // must be present, exactly once, at the end of the list. We don't care
                        // about any others because we don't interpret the message body.
                        fold(
                            reader,
                            &mut comma_separated(transfer_codings.clone(), |value| {
                                body.update(if value.is_some() {
                                    Body::Chunked
                                } else {
                                    Body::None
                                })
                            }),
                        )
                        .await?;
                        if !matches!(body, Body::Chunked) {
                            return Err(FramingError::BadSyntax);
                        }
                    }

                    // https://tools.ietf.org/html/rfc7230#section-6.1
                    Header::Connection => {
                        fold(
                            reader,
                            &mut comma_separated(keepalive_options.clone(), |value| {
                                if let Some(keepalive) = value {
                                    persistent = keepalive;
                                }
                                Ok(())
                            }),
                        )
                        .await?;
                    }

                    // https://tools.ietf.org/html/rfc7232#section-3.2
                    Header::IfNoneMatch => {
                        fold(
                            reader,
                            &mut comma_separated(etags.clone(), |value| {
                                // We only care about ETags that match some representation we currently
                                // have for this resource.
                                if let Some(value) = value {
                                    if_none_match.insert(value);
                                }
                                Ok(())
                            }),
                        )
                        .await?;
                    }

                    // https://tools.ietf.org/html/rfc7232#section-3.3
                    Header::IfModifiedSince => {
                        if_modified_since = fold(reader, &mut http_date()).await?
                    }

                    // https://tools.ietf.org/html/rfc7233#section-3.1
                    Header::Range => fold(reader, &mut byte_ranges(&mut range)).await?,

                    // https://tools.ietf.org/html/rfc7233#section-3.2
                    Header::IfRange => {
                        if_range = Some(fold(reader, &mut parse_if_range(etags.clone())).await?)
                    }
                }
                continue;
            }
        }

        fold(reader, &mut skip_line()).await?;
    }

    match body {
        Body::None => {}
        Body::Length(n) => fold(reader, &mut ok(skip(n))).await?,
        Body::Chunked => fold(reader, &mut chunked()).await?,
    }

    Ok(persistent)
}

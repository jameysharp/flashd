use futures::future::poll_fn;
use pin_project::pin_project;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::AsyncBufRead;

pub type ParseResult<V> = Result<Option<(usize, V)>, FramingError>;

pub async fn parse<R, F, V>(mut reader: Pin<&mut R>, mut f: F) -> std::io::Result<V>
where
    R: AsyncBufRead,
    F: FnMut(&[u8]) -> std::io::Result<(usize, V)>,
{
    poll_fn(|cx| {
        match reader.as_mut().poll_fill_buf(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(buf) => Poll::Ready(match buf.and_then(&mut f) {
                Err(e) => Err(e),
                Ok((consumed, v)) => {
                    reader.as_mut().consume(consumed);
                    Ok(v)
                }
            }),
        }
    }).await
}

macro_rules! stateful {
    ($start:expr => impl $(< $($typevar:ident),* >)? { $(
        for $state:ident
        ( $($arg:ident : $argty:ty),* )
        $(let $var:ident : $varty:ty = $init:expr ;)*
        match $parser:expr => $parseout:ty { $($pat:pat => $next:expr),* $(,)? }
    )* }) => {
        enum State $(<$($typevar),*>)? {
            None,
            $($state {
                $($arg: $argty,)*
                $($var: $varty,)*
            },)*
        }

        $(#[allow(non_snake_case)]
        let $state = move |$($arg: $argty),*| {
            $(let $var = $init;)*
            State::$state {
                $($arg,)*
                $($var,)*
            }
        };)*

        let mut state = $start;
        move |buf| {
            let mut consumed = 0;
            let result = loop {
                let rest = &buf[consumed..];
                if rest.is_empty() {
                    return Ok(None);
                }

                state = match ::std::mem::replace(&mut state, State::None) {
                    State::None => unreachable!(),
                    $(State::$state { $(mut $arg,)* $(mut $var,)* } => {
                        let result = match $parser(rest) {
                            Ok(Some((sub, value))) => {
                                let value: $parseout = value;
                                consumed += sub;
                                value
                            }
                            Ok(None) => {
                                state = State::$state { $($arg,)* $($var,)* };
                                return Ok(None);
                            }
                            Err(e) => return Err(e),
                        };
                        match result {
                            $($pat => $next),*
                        }
                    })*
                };
            };
            state = State::None;
            Ok(Some((consumed, result)))
        }
    };
}

pub fn fold<'a, T, F, V>(reader: &'a mut T, consumer: &'a mut F) -> Fold<'a, T, F>
where
    F: FnMut(&[u8]) -> ParseResult<V>,
    T: AsyncBufRead + Unpin,
{
    Fold { reader, consumer }
}

#[derive(Clone, Copy)]
pub enum FramingError {
    End,
    BadSyntax,
    BadVersion,
}

#[pin_project]
pub struct Fold<'a, T, F> {
    #[pin]
    reader: &'a mut T,
    consumer: &'a mut F,
}

impl<'a, T, F, V> Future for Fold<'a, T, F>
where
    F: FnMut(&[u8]) -> ParseResult<V>,
    T: AsyncBufRead + Unpin,
{
    type Output = Result<V, FramingError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        while let Poll::Ready(result) = this.reader.as_mut().poll_fill_buf(cx) {
            match result {
                Ok(buf) if !buf.is_empty() => {
                    let (consumed, value) = match (this.consumer)(buf) {
                        Ok(Some((consumed, value))) => (consumed, Some(value)),
                        Ok(None) => (buf.len(), None),
                        Err(e) => return Poll::Ready(Err(e)),
                    };
                    this.reader.as_mut().consume(consumed);
                    if let Some(value) = value {
                        return Poll::Ready(Ok(value));
                    }
                }
                _ => return Poll::Ready(Err(FramingError::End)),
            }
        }
        return Poll::Pending;
    }
}

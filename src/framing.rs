use async_compat::CompatExt;
use futures::io::AsyncBufReadExt;
use std::task::Poll;
use tokio::io::AsyncBufRead;

pub type ParseResult<V> = Poll<Result<(usize, V), FramingError>>;

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
                    return Poll::Pending;
                }

                state = match ::std::mem::replace(&mut state, State::None) {
                    State::None => unreachable!(),
                    $(State::$state { $(mut $arg,)* $(mut $var,)* } => {
                        let result = match $parser(rest) {
                            Poll::Ready(Ok((sub, value))) => {
                                let value: $parseout = value;
                                consumed += sub;
                                value
                            }
                            Poll::Pending => {
                                state = State::$state { $($arg,)* $($var,)* };
                                return Poll::Pending;
                            }
                            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                        };
                        match result {
                            $($pat => $next),*
                        }
                    })*
                };
            };
            state = State::None;
            Poll::Ready(Ok((consumed, result)))
        }
    };
}

pub async fn fold<T, F, V>(reader: &mut T, consumer: &mut F) -> Result<V, FramingError>
where
    F: FnMut(&[u8]) -> ParseResult<V>,
    T: AsyncBufRead + Unpin,
{
    let mut reader = reader.compat_mut();
    while let Ok(buf) = reader.fill_buf().await {
        if buf.is_empty() {
            break;
        }
        let (consumed, value) = match consumer(buf) {
            Poll::Ready(Ok((consumed, value))) => (consumed, Some(value)),
            Poll::Pending => (buf.len(), None),
            Poll::Ready(Err(e)) => return Err(e),
        };
        reader.consume_unpin(consumed);
        if let Some(value) = value {
            return Ok(value);
        }
    }
    Err(FramingError::End)
}

#[derive(Clone, Copy)]
pub enum FramingError {
    End,
    BadSyntax,
    BadVersion,
}

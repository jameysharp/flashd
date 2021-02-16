use async_compat::CompatExt;
use futures::io::AsyncBufReadExt;
use tokio::io::AsyncBufRead;

pub type ParseResult<V> = Result<Option<(usize, V)>, FramingError>;

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
        let (consumed, value) = match consumer(buf)? {
            Some((consumed, value)) => (consumed, Some(value)),
            None => (buf.len(), None),
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

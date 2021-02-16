use async_compat::CompatExt;
use futures::io::AsyncBufReadExt;
use futures::pin_mut;
use std::future::Future;
use std::task::Poll;
use tokio::io::AsyncBufRead;

use crate::parsing::run;

pub type ParseResult<V> = Poll<(usize, V)>;

pub async fn fold<T, F, V>(reader: &mut T, consumer: F) -> Result<V, FramingError>
where
    F: Future<Output = Result<V, FramingError>>,
    T: AsyncBufRead + Unpin,
{
    pin_mut!(consumer);
    let mut reader = reader.compat_mut();
    while let Ok(mut buf) = reader.fill_buf().await {
        let initial_len = buf.len();
        if initial_len == 0 {
            break;
        }
        let (consumed, value) = match run(&mut buf, consumer.as_mut()) {
            Poll::Ready(value) => (initial_len - buf.len(), Some(value)),
            Poll::Pending => (initial_len, None),
        };
        reader.consume_unpin(consumed);
        if let Some(value) = value {
            return value;
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

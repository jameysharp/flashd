use futures::future::poll_fn;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

pub fn run<F, V>(buf: &mut &[u8], f: Pin<&mut F>) -> Poll<V>
where
    F: Future<Output = V>,
{
    let waker = unsafe {
        let ptr = buf as *mut &[u8] as *mut () as *const ();
        Waker::from_raw(RawWaker::new(ptr, &VTABLE))
    };
    let mut cx = Context::from_waker(&waker);
    f.poll(&mut cx)
}

fn no_clone(_: *const ()) -> RawWaker {
    panic!("can't clone mutable reference to buffer");
}

fn no_wake(_: *const ()) {
    panic!("can't wake buffer: just poll the parser when you have more input");
}

fn no_drop(_: *const ()) {
    // no-op, a borrow doesn't need cleanup at drop time
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(no_clone, no_wake, no_wake, no_drop);

unsafe fn from_raw_waker<T>(waker: &Waker, expected: &'static RawWakerVTable) -> *const T {
    struct RawGuts {
        data: *const (),
        vtable: &'static RawWakerVTable,
    }
    struct Guts {
        raw: RawGuts,
    }

    let guts: Guts = std::mem::transmute_copy(waker);
    assert!(std::ptr::eq(guts.raw.vtable, expected));
    guts.raw.data as *const T
}

pub fn with_buf<F, V>(mut f: F) -> impl Future<Output = V>
where
    F: FnMut(&[u8]) -> Poll<(usize, V)>,
{
    poll_fn(move |cx| {
        let waker = cx.waker();
        let buf = unsafe {
            let ptr: *const &[u8] = from_raw_waker(waker, &VTABLE);
            (ptr as *mut &[u8]).as_mut().unwrap()
        };
        if !buf.is_empty() {
            if let Poll::Ready((consumed, v)) = f(buf) {
                *buf = &buf[consumed..];
                Poll::Ready(v)
            } else {
                Poll::Pending
            }
        } else {
            Poll::Pending
        }
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use futures::pin_mut;

    #[derive(Debug, Eq, PartialEq)]
    struct SyntaxError;
    async fn newline() -> Result<bool, SyntaxError> {
        if !expect(b'\r').await {
            Ok(expect(b'\n').await)
        } else if expect(b'\n').await {
            Ok(true)
        } else {
            Err(SyntaxError)
        }
    }

    fn expect(b: u8) -> impl Future<Output = bool> {
        with_buf(move |buf| {
            let matched = buf[0] == b;
            Poll::Ready((matched as usize, matched))
        })
    }

    #[test]
    fn match_all() {
        let mut examples = [
            (0, &b"\r"[..], Poll::Pending),
            (0, b"\r\n", Poll::Ready(Ok(true))),
            (1, b"\rA", Poll::Ready(Err(SyntaxError))),
            (0, b"\n", Poll::Ready(Ok(true))),
            (1, b"\n\r", Poll::Ready(Ok(true))),
            (2, b"A\n", Poll::Ready(Ok(false))),
        ];
        for (unused, example, expected) in examples.iter_mut() {
            let parser = newline();
            pin_mut!(parser);
            let result = run(example, parser);
            assert_eq!(example.len(), *unused);
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn match_one_byte() {
        let mut examples = [
            (0, &b"\r"[..], Poll::Pending),
            (0, b"\r\n", Poll::Ready(Ok(true))),
            (1, b"\rA", Poll::Ready(Err(SyntaxError))),
            (0, b"\n", Poll::Ready(Ok(true))),
            (1, b"\n\r", Poll::Ready(Ok(true))),
            (2, b"A\n", Poll::Ready(Ok(false))),
        ];
        for (unused, example, expected) in examples.iter_mut() {
            let parser = newline();
            pin_mut!(parser);
            let mut bytes = example.iter();
            while let Some(b) = bytes.next() {
                let mut b = std::slice::from_ref(b);
                if let Poll::Ready(result) = run(&mut b, parser.as_mut()) {
                    assert_eq!(b.len() + bytes.as_slice().len(), *unused);
                    assert_eq!(Poll::Ready(result), *expected);
                    return;
                }
            }
            assert_eq!(0, *unused);
            assert_eq!(Poll::Pending, *expected);
        }
    }
}

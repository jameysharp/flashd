use std::cmp::Ordering;
use std::marker::PhantomData;
use std::task::Poll;

use crate::framing::ParseResult;

pub trait Comparator {
    type Item;
    fn cmp(&self, l: &[Self::Item], r: &[Self::Item]) -> Ordering;
}

#[derive(Clone, Copy)]
pub struct Natural<T>(PhantomData<T>);

impl<T> Natural<T> {
    pub const ORDER: Self = Natural(PhantomData);
}

impl<T: Ord> Comparator for Natural<T> {
    type Item = T;
    fn cmp(&self, l: &[T], r: &[T]) -> Ordering {
        l.cmp(r)
    }
}

#[derive(Clone, Copy)]
pub struct CaseInsensitiveASCII;

impl Comparator for CaseInsensitiveASCII {
    type Item = u8;
    fn cmp(&self, l: &[u8], r: &[u8]) -> Ordering {
        let l = l.iter().map(|&b| b.to_ascii_lowercase());
        let r = r.iter().map(|&b| b.to_ascii_lowercase());
        l.cmp(r)
    }
}

#[derive(Clone)]
pub struct Matcher<'a, C, V>
where
    C: Comparator,
{
    comparator: C,
    patterns: &'a [(&'a [C::Item], V)],
    offset: usize,
}

fn is_strictly_sorted_by<'a, C, I>(iter: I, cmp: &C) -> bool
where
    C: Comparator,
    C::Item: 'a,
    I: Iterator<Item = &'a [C::Item]> + Clone,
{
    let before = iter.clone();
    let after = iter.skip(1);
    before
        .zip(after)
        .all(|(a, b)| cmp.cmp(a, b) == Ordering::Less)
}

impl<'a, C, V> Matcher<'a, C, V>
where
    C: Comparator,
    V: Copy,
{
    /// Prepares to match against the provided patterns, using a specified implementation of
    /// [Comparator] to define the sort order on patterns and inputs. The patterns must already be
    /// sorted and unique with respect to the given comparator.
    pub fn new(comparator: C, patterns: &'a [(&'a [C::Item], V)]) -> Self {
        debug_assert!(is_strictly_sorted_by(
            patterns.iter().map(|(pattern, _)| *pattern),
            &comparator
        ));

        Matcher {
            comparator,
            patterns,
            offset: 0,
        }
    }

    /// Matches the input against the remaining unchecked portions of the patterns. Any patterns
    /// which can't match the input seen so far are removed from future consideration. The portions
    /// which have just been checked are also ignored in future calls.
    pub fn push(&mut self, input: &[C::Item]) -> ParseResult<Option<V>> {
        Poll::Pending
    }
}

/// Prepares to match against the provided patterns, using a specified implementation of
/// [Comparator] to define the sort order on patterns and inputs. The patterns must already be
/// sorted and unique with respect to the given comparator.
pub fn matcher<'a, C, V>(
    comparator: C,
    patterns: &'a [(&'a [C::Item], V)],
) -> impl FnMut(&[C::Item]) -> ParseResult<Option<V>> + Clone + 'a
where
    C: Comparator + Clone + 'a,
    C::Item: Clone,
    V: Copy,
{
    let mut state = Matcher::new(comparator, patterns);
    move |input| state.push(input)
}

/*
fn find_edge<V>(
    buffer: &[u8],
    patterns: &[(&[u8], V)],
    offset: usize,
    tiebreaker: Ordering,
) -> usize {
    let idx = patterns.binary_search_by(|(pattern, _)| {
        if buffer.len() > pattern.len() {
            pattern[offset..].cmp(&&buffer[offset..pattern.len()])
        } else {
            pattern[offset..buffer.len()]
                .cmp(&buffer[offset..])
                .then(Ordering::Greater)
        }
    });
    match idx {
        Ok(v) => v,
        Err(v) => v,
    }
}

let mut start = 0;

loop {
    let lo = find_edge(&self.buffer, patterns, start, Ordering::Greater);
    patterns = &patterns[lo..];

    let hi = find_edge(&self.buffer, patterns, start, Ordering::Less);
    patterns = &patterns[..hi];

    match patterns {
        [] => return Poll::Pending,
        [(pattern, value)] if pattern.len() <= self.buffer.len() => {
            self.buffer.advance(pattern.len());
            return Poll::Ready(Ok(*value));
        }
        _ => {}
    }

    start = self.buffer.len();
    self.fill_buf().await?;
}
*/

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn partial_match() {
        let mut m = Matcher::new(
            Natural::ORDER,
            &[
                (b"aaa", 1),
                (b"baa", 2),
                (b"bbb", 3),
                (b"bcc", 4),
                (b"ccc", 5),
            ],
        );

        assert_eq!(m.push(b"b"), None);
        assert_eq!(m.patterns.len(), 3);
        assert_eq!(m.offset, 1);

        assert_eq!(m.push(b"b"), None);
        assert_eq!(m.patterns.len(), 1);
        assert_eq!(m.offset, 2);

        assert_eq!(m.push(b"b"), Some((1, Some(3))));
    }

    #[test]
    fn equal_length() {
        let mut m = Matcher::new(
            Natural::ORDER,
            &[
                (b"aaa", 1),
                (b"baa", 2),
                (b"bbb", 3),
                (b"bcc", 4),
                (b"ccc", 5),
            ],
        );
        assert_eq!(m.push(b"bbb"), Some((3, Some(3))));
    }

    #[test]
    fn partial_match_common_prefix() {
        let mut m = Matcher::new(Natural::ORDER, &[(b"a", 1), (b"aa", 2), (b"aaa", 3)]);

        assert_eq!(m.push(b"a"), None);
        assert_eq!(m.patterns.len(), 3);
        assert_eq!(m.offset, 1);

        assert_eq!(m.clone().push(b"z"), Some((0, Some(1))));

        assert_eq!(m.push(b"a"), None);
        assert_eq!(m.patterns.len(), 2);
        assert_eq!(m.offset, 2);

        assert_eq!(m.clone().push(b"z"), Some((0, Some(2))));

        assert_eq!(m.push(b"a"), Some((1, Some(3))));
    }

    #[test]
    fn long_input() {
        let mut m = Matcher::new(
            Natural::ORDER,
            &[
                (b"aaa", 1),
                (b"baa", 2),
                (b"bbb", 3),
                (b"bcc", 4),
                (b"ccc", 5),
            ],
        );
        assert_eq!(m.push(b"bbbz"), Some((3, Some(3))));
    }

    #[test]
    fn case_insensitive() {
        let mut m = Matcher::new(
            CaseInsensitiveASCII,
            &[(b"aBc", 1), (b"AbD", 2), (b"ABE", 3)],
        );
        assert_eq!(m.clone().push(b"abd"), Some((3, Some(2))));
        assert_eq!(m.clone().push(b"aBd"), Some((3, Some(2))));
        assert_eq!(m.clone().push(b"ABD"), Some((3, Some(2))));
    }
}

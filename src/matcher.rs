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
        let idx = self
            .patterns
            .iter()
            .position(|(pattern, _)| {
                let common = input.len().min(pattern.len() - self.offset);
                self.comparator.cmp(
                    &pattern[self.offset..self.offset + common],
                    &input[..common],
                ) == Ordering::Equal
            })
            .unwrap_or(self.patterns.len());
        self.patterns = &self.patterns[idx..];

        let idx = self
            .patterns
            .iter()
            .position(|(pattern, _)| {
                let common = input.len().min(pattern.len() - self.offset);
                self.comparator.cmp(
                    &pattern[self.offset..self.offset + common],
                    &input[..common],
                ) != Ordering::Equal
            })
            .unwrap_or(self.patterns.len());
        self.patterns = &self.patterns[..idx];

        match self.patterns {
            [] => Poll::Ready((0, None)),
            [(pattern, value)] if pattern.len() - self.offset <= input.len() => {
                Poll::Ready((pattern.len() - self.offset, Some(*value)))
            }
            _ => {
                self.offset += input.len();
                Poll::Pending
            }
        }
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

        assert_eq!(m.push(b"b"), Poll::Pending);
        assert_eq!(m.patterns.len(), 3);
        assert_eq!(m.offset, 1);

        assert_eq!(m.push(b"b"), Poll::Pending);
        assert_eq!(m.patterns.len(), 1);
        assert_eq!(m.offset, 2);

        assert_eq!(m.push(b"b"), Poll::Ready((1, Some(3))));
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
        assert_eq!(m.push(b"bbb"), Poll::Ready((3, Some(3))));
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
        assert_eq!(m.push(b"bbbz"), Poll::Ready((3, Some(3))));
    }

    #[test]
    fn case_insensitive() {
        let mut m = Matcher::new(
            CaseInsensitiveASCII,
            &[(b"aBc", 1), (b"AbD", 2), (b"ABE", 3)],
        );
        assert_eq!(m.clone().push(b"abd"), Poll::Ready((3, Some(2))));
        assert_eq!(m.clone().push(b"aBd"), Poll::Ready((3, Some(2))));
        assert_eq!(m.clone().push(b"ABD"), Poll::Ready((3, Some(2))));
    }
}

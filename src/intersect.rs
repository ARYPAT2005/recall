//! Intersecting sorted lists of doc ids.

use crate::DocId;
use std::cmp::Ordering;

/// Which intersection algorithm to use. `search` uses Adaptive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Merge,
    Gallop,
    Adaptive,
}

/// Adaptive gallops once one list is this many times longer (measured in bench.rs).
pub const GALLOP_RATIO: usize = 16;

impl Strategy {
    pub const ALL: [Strategy; 3] = [Strategy::Merge, Strategy::Gallop, Strategy::Adaptive];

    /// The algorithm Adaptive actually picks for lists of these lengths.
    pub fn resolve(self, a_len: usize, b_len: usize) -> Strategy {
        let (shorter, longer) = (a_len.min(b_len), a_len.max(b_len));
        match self {
            Strategy::Adaptive if longer >= shorter.saturating_mul(GALLOP_RATIO) => Strategy::Gallop,
            Strategy::Adaptive => Strategy::Merge,
            fixed => fixed,
        }
    }
}

/// Doc ids found in both sorted lists.
pub fn intersect_with(a: &[DocId], b: &[DocId], strategy: Strategy) -> Vec<DocId> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match strategy.resolve(small.len(), large.len()) {
        Strategy::Merge => intersect_merge(small, large),
        _ => intersect_gallop(small, large),
    }
}

/// Index of the first element >= id.
pub(crate) fn gallop_to(s: &[DocId], id: DocId) -> usize {
    gallop_by(s, |&x| x < id)
}

/// Galloping search: check positions 1, 2, 4, 8... then binary search the last gap.
pub(crate) fn gallop_by<T>(s: &[T], mut before: impl FnMut(&T) -> bool) -> usize {
    let mut hi = 1;
    while hi < s.len() && before(&s[hi]) {
        hi *= 2;
    }
    let lo = hi / 2;
    let end = (hi + 1).min(s.len());
    lo + s[lo..end].partition_point(before)
}

/// O(m log(n/m)). Fast when one list is much longer than the other.
pub fn intersect_gallop(small: &[DocId], large: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(small.len());
    let mut base = 0;
    for &id in small {
        base += gallop_to(&large[base..], id);
        if base == large.len() {
            break;
        }
        if large[base] == id {
            out.push(id);
            base += 1;
        }
    }
    out
}

/// O(m + n). Fast when the lists are about the same length.
pub fn intersect_merge(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_keeps_common_ids() {
        for s in Strategy::ALL {
            assert_eq!(intersect_with(&[1, 3, 5, 8, 12], &[3, 4, 8, 20], s), vec![3, 8]);
            assert_eq!(intersect_with(&[1, 2], &[3, 4], s), Vec::<DocId>::new());
            assert_eq!(intersect_with(&[], &[1, 2], s), Vec::<DocId>::new());
            assert_eq!(intersect_with(&[1, 2], &[], s), Vec::<DocId>::new());
        }
    }

    #[test]
    fn gallop_handles_edges_of_the_long_list() {
        let large: Vec<DocId> = (0..1000).map(|i| i * 3).collect();
        assert_eq!(intersect_gallop(&[0], &large), vec![0]);
        assert_eq!(intersect_gallop(&[2997], &large), vec![2997]);
        assert_eq!(intersect_gallop(&[5000], &large), Vec::<DocId>::new());
        assert_eq!(intersect_gallop(&[1, 2, 3], &large), vec![3]);
        assert_eq!(intersect_gallop(&[3, 6, 9], &large), vec![3, 6, 9]);
    }

    #[test]
    fn gallop_matches_merge_on_random_lists() {
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let universe = 1 + next() % 5000;
            let (da, db) = (1 + next() % 1000, 1 + next() % 1000);
            let mut list = |density: u64| -> Vec<DocId> {
                (0..universe as DocId).filter(|_| next() % 1000 < density).collect()
            };
            let a = list(da);
            let b = list(db);
            let expected = intersect_merge(&a, &b);
            assert_eq!(intersect_with(&a, &b, Strategy::Gallop), expected);
            assert_eq!(intersect_with(&b, &a, Strategy::Gallop), expected);
            assert_eq!(intersect_with(&a, &b, Strategy::Adaptive), expected);
        }
    }
}

//! Intersecting sorted doc-id lists: linear merge, galloping search, and the
//! rule for choosing between them.
//!
//! The slice functions here are the reference implementations. Compressed
//! posting lists (postings.rs) run the same two algorithms over blocks, and
//! their tests check them against these.

use crate::DocId;
use std::cmp::Ordering;

/// How two posting lists get intersected. `Adaptive` is what `search` uses;
/// the other two exist so the benchmark can measure each algorithm alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Merge,
    Gallop,
    Adaptive,
}

/// Adaptive switches from merge to galloping once the longer list is at least
/// this many times the shorter one. Chosen from the ratio sweep in bench.rs,
/// and it depends on BLOCK: with 128-entry blocks the crossover moved to
/// between 16x and 32x; with 64 it's back between 8x and 16x.
pub const GALLOP_RATIO: usize = 16;

impl Strategy {
    pub const ALL: [Strategy; 3] = [Strategy::Merge, Strategy::Gallop, Strategy::Adaptive];

    /// The algorithm this strategy runs for lists of these lengths: Adaptive
    /// resolves to Merge or Gallop, the others to themselves.
    pub fn resolve(self, a_len: usize, b_len: usize) -> Strategy {
        let (shorter, longer) = (a_len.min(b_len), a_len.max(b_len));
        match self {
            Strategy::Adaptive if longer >= shorter.saturating_mul(GALLOP_RATIO) => Strategy::Gallop,
            Strategy::Adaptive => Strategy::Merge,
            fixed => fixed,
        }
    }
}

/// Documents present in both sorted, deduped lists, using `strategy`.
pub fn intersect_with(a: &[DocId], b: &[DocId], strategy: Strategy) -> Vec<DocId> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match strategy.resolve(small.len(), large.len()) {
        Strategy::Merge => intersect_merge(small, large),
        _ => intersect_gallop(small, large),
    }
}

/// Index of the first element of `s` that is >= `id`.
pub(crate) fn gallop_to(s: &[DocId], id: DocId) -> usize {
    gallop_by(s, |&x| x < id)
}

/// Index of the first element for which `before` is false, where `before`
/// is true for a prefix of `s`. Probes s[1], s[2], s[4]... until one is past
/// the boundary, then binary-searches that bracket, so the cost is O(log k)
/// where k is the answer: a nearby target is cheap.
pub(crate) fn gallop_by<T>(s: &[T], mut before: impl FnMut(&T) -> bool) -> usize {
    // Double `hi` while s[hi] is still before the boundary. When it stops,
    // the answer lies in s[hi/2 ..= hi].
    let mut hi = 1;
    while hi < s.len() && before(&s[hi]) {
        hi *= 2;
    }
    let lo = hi / 2;
    let end = (hi + 1).min(s.len());
    lo + s[lo..end].partition_point(before)
}

/// Galloping (exponential) search: for each id in `small`, gallop forward in
/// `large` from the last match. Cost is O(m log(n/m)) instead of O(m + n), so
/// "zyzzyva AND the" skips most of the 90K-entry list rather than walking it.
/// Loses to merge when the lists are similar in length: every step pays for a
/// binary search where merge would just bump a cursor.
pub fn intersect_gallop(small: &[DocId], large: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(small.len());
    // Everything in large[..base] is already known to be < the current id.
    let mut base = 0;
    for &id in small {
        base += gallop_to(&large[base..], id);
        if base == large.len() {
            break; // every remaining id in small is past the end of large
        }
        if large[base] == id {
            out.push(id);
            base += 1;
        }
    }
    out
}

/// Two cursors walk in lockstep and the smaller one advances, so this is
/// O(len(a) + len(b)) with no hashing - it only works because add_document
/// keeps every list sorted.
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
        // First element, last element, past the end, before the start.
        assert_eq!(intersect_gallop(&[0], &large), vec![0]);
        assert_eq!(intersect_gallop(&[2997], &large), vec![2997]);
        assert_eq!(intersect_gallop(&[5000], &large), Vec::<DocId>::new());
        assert_eq!(intersect_gallop(&[1, 2, 3], &large), vec![3]);
        // Consecutive matches must not skip each other.
        assert_eq!(intersect_gallop(&[3, 6, 9], &large), vec![3, 6, 9]);
    }

    #[test]
    fn gallop_matches_merge_on_random_lists() {
        // xorshift, fixed seed: same cases every run.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            // Vary density independently so ratios range from 1:1 to ~1:1000.
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

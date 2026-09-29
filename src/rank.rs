//! BM25 scoring pieces and top-k selection.

use crate::DocId;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// BM25's standard parameters. K1 limits how much repeating a term can add
/// (the 10th "rust" in a doc is worth far less than the 1st). B sets how
/// strongly long documents are penalized for matching by sheer length.
pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// Inverse document frequency: rare terms count for more. This is the Lucene
/// form, ln(1 + (N - df + 0.5) / (df + 0.5)), which stays positive even for a
/// term in every document (the textbook form goes negative there).
pub fn idf(num_docs: usize, df: usize) -> f32 {
    let (n, df) = (num_docs as f32, df as f32);
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// A scored document.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub doc: DocId,
    pub score: f32,
}

/// Ordered best first: `a < b` means a ranks above b. Higher score wins, and
/// ties go to the lower doc id so results come out the same every run.
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.doc.cmp(&other.doc))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Hit {}

/// The k best hits, best first, in O(n log k) instead of sorting all n.
///
/// BinaryHeap is a max-heap, and Hit orders best first, so the heap's top is
/// the *worst* of the k kept so far. Each new hit either loses to it - one
/// comparison, the common case once the heap has filled with good hits - or
/// replaces it and sifts down in O(log k).
///
/// When k covers every hit there is nothing to select: pushing them all
/// through the heap would just be a slow heap sort, so sort directly.
pub fn top_k(hits: impl IntoIterator<Item = Hit>, k: usize) -> Vec<Hit> {
    let hits = hits.into_iter();
    if hits.size_hint().1.is_some_and(|n| n <= k) {
        let mut all: Vec<Hit> = hits.collect();
        all.sort_unstable();
        return all;
    }
    let mut heap = BinaryHeap::new();
    if k == 0 {
        return Vec::new();
    }
    for hit in hits {
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(mut worst) = heap.peek_mut() {
            if hit < *worst {
                *worst = hit;
            }
        }
    }
    heap.into_sorted_vec()
}

/// Ranked results: how many documents matched, and the best of them.
pub struct Ranked {
    pub matched: usize,
    pub hits: Vec<Hit>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_equals_full_sort() {
        let hits: Vec<Hit> = (0..500)
            .map(|i| Hit { doc: i, score: ((i * 7919) % 97) as f32 })
            .collect();
        let mut sorted = hits.clone();
        sorted.sort();
        for k in [1, 5, 10, 97, 500, 1000] {
            let got: Vec<DocId> = top_k(hits.iter().copied(), k).iter().map(|h| h.doc).collect();
            let want: Vec<DocId> = sorted.iter().take(k).map(|h| h.doc).collect();
            assert_eq!(got, want, "k = {k}");
        }
    }
}

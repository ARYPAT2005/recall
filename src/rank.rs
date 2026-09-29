//! BM25 scoring and top-k selection.

use crate::DocId;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Standard BM25 parameters.
pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// Lucene's version of idf, which never goes negative.
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

/// Best first: higher score wins, ties go to the lower doc id.
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

/// The k best hits, best first. Keeps a heap of size k instead of sorting everything.
pub fn top_k(hits: impl IntoIterator<Item = Hit>, k: usize) -> Vec<Hit> {
    let hits = hits.into_iter();
    // If k covers every hit, just sort.
    if hits.size_hint().1.is_some_and(|n| n <= k) {
        let mut all: Vec<Hit> = hits.collect();
        all.sort_unstable();
        return all;
    }
    // The heap's top is the worst hit kept, so most new hits lose one comparison.
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

/// How many documents matched, and the best ones.
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

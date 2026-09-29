//! Recall: a full-text search engine, built from scratch.
//!
//! This is the *library*. The binaries in `src/bin/` and `src/main.rs` all
//! import it. Keeping the engine in a lib (instead of one big main.rs) is what
//! makes sharding possible later: a shard is just this struct in its own
//! process.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// A document's identity is just its position in `doc_names`.
///
/// u32, not usize: 4 bytes instead of 8. Posting lists are the bulk of the
/// index, so this literally halves the memory of the largest structure in the
/// program. It caps us at ~4.3 billion documents, which is fine.
pub type DocId = u32;

/// "Machine Learning, fast!" -> ["machine", "learning", "fast"]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// How two posting lists get intersected. `Adaptive` is what `search` uses;
/// the other two exist so the benchmark can measure each algorithm alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Merge,
    Gallop,
    Adaptive,
}

/// Adaptive switches from merge to galloping once the longer list is at least
/// this many times the shorter one. Chosen from the ratio sweep in bench.rs.
pub const GALLOP_RATIO: usize = 16;

/// Documents present in both sorted, deduped lists, choosing the algorithm
/// from the length ratio.
pub fn intersect(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    intersect_with(a, b, Strategy::Adaptive)
}

pub fn intersect_with(a: &[DocId], b: &[DocId], strategy: Strategy) -> Vec<DocId> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match strategy {
        Strategy::Merge => intersect_merge(small, large),
        Strategy::Gallop => intersect_gallop(small, large),
        Strategy::Adaptive => {
            if large.len() >= small.len().saturating_mul(GALLOP_RATIO) {
                intersect_gallop(small, large)
            } else {
                intersect_merge(small, large)
            }
        }
    }
}

/// Galloping (exponential) search: for each id in `small`, probe `large` at
/// offsets 1, 2, 4, 8... past the last match until we overshoot, then binary
/// search inside that bracket. Cost is O(m log(n/m)) instead of O(m + n), so
/// "zyzzyva AND the" skips most of the 90K-entry list rather than walking it.
/// Loses to merge when the lists are similar in length: every step pays for a
/// binary search where merge would just bump a cursor.
pub fn intersect_gallop(small: &[DocId], large: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(small.len());
    // Everything in large[..base] is already known to be < the current id.
    let mut base = 0;
    for &id in small {
        let rest = &large[base..];

        // Double `hi` while rest[hi] is still too small. When it stops, id
        // (if present) lies in rest[hi/2 ..= hi].
        let mut hi = 1;
        while hi < rest.len() && rest[hi] < id {
            hi *= 2;
        }
        let lo = hi / 2;
        let end = (hi + 1).min(rest.len());
        base += lo + rest[lo..end].partition_point(|&x| x < id);

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

#[derive(Default)]
pub struct Index {
    /// The inverted index: term -> ascending list of documents containing it.
    pub postings: HashMap<String, Vec<DocId>>,
    /// doc_names[id] -> the filename or title
    pub doc_names: Vec<String>,
    /// doc_lens[id] -> token count. Unused today; BM25 needs it in step 5.
    pub doc_lens: Vec<u32>,
}

impl Index {
    pub fn new() -> Self {
        Index::default()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_names.len()
    }

    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    /// Total entries across all posting lists - the real size driver.
    pub fn num_postings(&self) -> usize {
        self.postings.values().map(|v| v.len()).sum()
    }

    /// Average document length in tokens. BM25 needs this.
    pub fn avg_doc_len(&self) -> f64 {
        if self.doc_lens.is_empty() {
            return 0.0;
        }
        self.doc_lens.iter().map(|&l| l as f64).sum::<f64>() / self.doc_lens.len() as f64
    }

    pub fn add_document(&mut self, name: String, text: &str) -> DocId {
        let id = self.doc_names.len() as DocId;
        let tokens = tokenize(text);

        for word in tokens.iter() {
            let list = self.postings.entry(word.clone()).or_default();
            // Docs arrive in ascending id order, so a duplicate can only be the
            // final element. This single check keeps every list deduped AND
            // sorted - which is exactly what the intersection in step 4 needs.
            if list.last() != Some(&id) {
                list.push(id);
            }
        }

        self.doc_names.push(name);
        self.doc_lens.push(tokens.len() as u32);
        id
    }

    /// The posting list for one term. Empty slice if the term is unknown.
    pub fn postings_for(&self, term: &str) -> &[DocId] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Boolean AND: documents containing every term in the query. The query
    /// goes through the same tokenizer as the documents, so "Machine
    /// Learning!" and "machine learning" are the same search.
    pub fn search(&self, query: &str) -> Vec<DocId> {
        self.search_with(query, Strategy::Adaptive)
    }

    pub fn search_with(&self, query: &str, strategy: Strategy) -> Vec<DocId> {
        let mut lists: Vec<&[DocId]> = tokenize(query)
            .iter()
            .map(|t| self.postings_for(t))
            .collect();
        if lists.is_empty() {
            return Vec::new();
        }

        // Shortest first: the result can never outgrow the shortest list, so
        // starting there bounds every later intersection by the rarest term
        // instead of dragging the longest list through each step.
        lists.sort_by_key(|l| l.len());

        if lists.len() == 1 {
            return lists[0].to_vec();
        }
        // Intersect the first pair straight from the index instead of copying
        // the shortest list just to intersect the copy.
        let mut result = intersect_with(lists[0], lists[1], strategy);
        for list in &lists[2..] {
            if result.is_empty() {
                break;
            }
            result = intersect_with(&result, list, strategy);
        }
        result
    }

    /// Index every .txt file in a directory. Good for small hand-written corpora.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> io::Result<Index> {
        let mut idx = Index::new();
        let mut paths: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect();
        paths.sort(); // stable doc ids across runs

        for path in paths {
            let text = fs::read_to_string(&path)?;
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            idx.add_document(name, &text);
        }
        Ok(idx)
    }

    /// Index a corpus file: one document per line, `title<TAB>body`.
    ///
    /// 100K separate files would mean 100K `open`/`close` syscalls and a lot of
    /// wasted inodes. One file read sequentially is dramatically faster and is
    /// how real corpora ship.
    pub fn from_corpus_file<P: AsRef<Path>>(path: P) -> io::Result<Index> {
        let mut idx = Index::new();
        let reader = BufReader::with_capacity(1 << 20, File::open(path)?);

        for line in reader.lines() {
            let line = line?;
            let (title, body) = match line.split_once('\t') {
                Some(pair) => pair,
                None => continue,
            };
            idx.add_document(title.to_string(), body);
        }
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Index {
        let mut idx = Index::new();
        idx.add_document("a".into(), "machine learning with rust");
        idx.add_document("b".into(), "machine shop");
        idx.add_document("c".into(), "deep learning, machine learning!");
        idx.add_document("d".into(), "rust learning");
        idx
    }

    const STRATEGIES: [Strategy; 3] = [Strategy::Merge, Strategy::Gallop, Strategy::Adaptive];

    #[test]
    fn intersect_keeps_common_ids() {
        for s in STRATEGIES {
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
            assert_eq!(intersect(&a, &b), expected);
        }
    }

    #[test]
    fn search_is_boolean_and() {
        let idx = sample();
        assert_eq!(idx.search("machine learning"), vec![0, 2]);
        assert_eq!(idx.search("rust learning machine"), vec![0]);
        assert_eq!(idx.search("machine"), vec![0, 1, 2]);
    }

    #[test]
    fn search_tokenizes_the_query() {
        let idx = sample();
        assert_eq!(idx.search("  Machine, LEARNING!\n"), vec![0, 2]);
        assert_eq!(idx.search("learning learning"), vec![0, 2, 3]);
    }

    #[test]
    fn unknown_term_or_empty_query_matches_nothing() {
        let idx = sample();
        assert!(idx.search("machine zebra").is_empty());
        assert!(idx.search("").is_empty());
        assert!(idx.search("!!!").is_empty());
    }
}

//! The single-threaded index: term dictionary, posting lists, document
//! metadata, and the query path. A shard is one of these.

use crate::hash::FxBuildHasher;
use crate::postings::{PostingList, NO_POSTINGS};
use crate::rank::{idf, top_k, Hit, Ranked, BM25_B, BM25_K1};
use crate::tokenize::{for_each_token, tokenize};
use crate::{DocId, Plan, PlanStep, PlanTerm, Strategy};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

#[derive(Default)]
pub struct Index {
    /// Term -> its id, an index into `lists`.
    term_ids: HashMap<String, u32, FxBuildHasher>,
    /// The inverted index: term id -> its posting list.
    lists: Vec<PostingList>,
    /// Scratch for add_document: the current doc's count for each term id,
    /// and which ids it has touched. Kept here so no doc allocates them.
    pending_tf: Vec<u32>,
    touched: Vec<u32>,
    /// doc_names[id] -> the filename or title
    pub doc_names: Vec<String>,
    /// doc_lens[id] -> token count, for BM25's length normalization.
    pub doc_lens: Vec<u32>,
    /// Sum of doc_lens, so avg_doc_len is O(1) on the query path.
    total_len: u64,
}

impl Index {
    pub fn new() -> Self {
        Index::default()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_names.len()
    }

    pub fn num_terms(&self) -> usize {
        self.lists.len()
    }

    /// Total entries across all posting lists - the real size driver.
    pub fn num_postings(&self) -> usize {
        self.lists.iter().map(|l| l.len()).sum()
    }

    /// Heap bytes held by all posting lists (not the term strings or table):
    /// (holding entries, allocated including unused capacity).
    pub fn posting_bytes(&self) -> (usize, usize) {
        self.lists
            .iter()
            .map(|l| l.heap_bytes())
            .fold((0, 0), |(u, a), (lu, la)| (u + lu, a + la))
    }

    /// Average document length in tokens. BM25 needs this.
    pub fn avg_doc_len(&self) -> f64 {
        if self.doc_lens.is_empty() {
            return 0.0;
        }
        self.total_len as f64 / self.doc_lens.len() as f64
    }

    /// Every term with its posting list, in no particular order.
    pub fn terms(&self) -> impl Iterator<Item = (&str, &PostingList)> {
        self.term_ids.iter().map(|(t, &id)| (t.as_str(), &self.lists[id as usize]))
    }

    pub fn add_document(&mut self, name: String, text: &str) -> DocId {
        let id = self.doc_names.len() as DocId;
        let Index { term_ids, lists, pending_tf, touched, .. } = self;
        let mut len = 0u32;

        // Count each term in this doc first, then append one (doc, tf) entry
        // per distinct term. Compressed lists are append-only, so a repeat
        // can't go back and bump a tf that's already been encoded.
        for_each_token(text, &mut String::new(), |word| {
            len += 1;
            // get first: entry() needs an owned String key, which would
            // allocate once per token even though nearly every term already
            // exists. Only a brand-new term pays for the allocation.
            let term = match term_ids.get(word) {
                Some(&term) => term,
                None => {
                    let term = lists.len() as u32;
                    term_ids.insert(word.to_owned(), term);
                    lists.push(PostingList::default());
                    pending_tf.push(0);
                    term
                }
            };
            let count = &mut pending_tf[term as usize];
            if *count == 0 {
                touched.push(term);
            }
            *count += 1;
        });
        for term in touched.drain(..) {
            let count = &mut pending_tf[term as usize];
            lists[term as usize].push(id, *count);
            *count = 0;
        }

        self.doc_names.push(name);
        self.doc_lens.push(len);
        self.total_len += len as u64;
        id
    }

    /// The posting list for one term, if any document contains it.
    pub fn postings_for(&self, term: &str) -> Option<&PostingList> {
        self.term_ids.get(term).map(|&id| &self.lists[id as usize])
    }

    /// Boolean AND: documents containing every term in the query, ascending.
    /// The query goes through the same tokenizer as the documents, so "Machine
    /// Learning!" and "machine learning" are the same search.
    pub fn search(&self, query: &str) -> Vec<DocId> {
        self.search_with(query, Strategy::Adaptive)
    }

    pub fn search_with(&self, query: &str, strategy: Strategy) -> Vec<DocId> {
        self.matching(query, strategy, None).1
    }

    /// The k most relevant documents containing every query term, by BM25.
    pub fn search_ranked(&self, query: &str, k: usize) -> Ranked {
        let (lists, docs) = self.matching(query, Strategy::Adaptive, None);
        self.rank(&lists, &docs, k)
    }

    /// Run a ranked query and report every decision the engine made: term
    /// order, each intersection's algorithm and why, and where time went.
    pub fn explain(&self, query: &str, k: usize) -> Plan {
        // Run it once and throw that plan away. The first run pays one-time
        // costs - cold posting lists, and the process's first clock read,
        // which on macOS resolves the timer symbol lazily (~13 µs) - that
        // would otherwise land on whichever step happens to go first.
        self.explain_once(query, k);
        self.explain_once(query, k)
    }

    fn explain_once(&self, query: &str, k: usize) -> Plan {
        let mut plan = Plan {
            query: query.to_owned(),
            num_docs: self.num_docs(),
            k,
            ..Plan::default()
        };
        let (lists, docs) = self.matching(query, Strategy::Adaptive, Some(&mut plan));
        let start = Instant::now();
        let ranked = self.rank(&lists, &docs, k);
        plan.rank_time = start.elapsed();
        plan.matched = ranked.matched;
        plan.hits = ranked.hits;
        plan
    }

    fn rank(&self, lists: &[&PostingList], docs: &[DocId], k: usize) -> Ranked {
        let scores = self.score(lists, docs);
        let hits = docs.iter().zip(&scores).map(|(&doc, &score)| Hit { doc, score });
        Ranked {
            matched: docs.len(),
            hits: top_k(hits, k),
        }
    }

    /// The query's posting lists (rarest first) and the docs in all of them.
    /// With `plan`, also records each decision and its timing - the only time
    /// this path reads the clock.
    fn matching(
        &self,
        query: &str,
        strategy: Strategy,
        mut plan: Option<&mut Plan>,
    ) -> (Vec<&PostingList>, Vec<DocId>) {
        let started = plan.is_some().then(Instant::now);
        let mut terms = tokenize(query);
        // "rust rust" is one term. Scoring it twice would double its weight.
        terms.sort_unstable();
        terms.dedup();

        // An unknown term gets an empty list instead of a special case: it
        // sorts first as the rarest term, and the intersection comes out
        // empty, which is exactly what AND means.
        let mut lists: Vec<(String, &PostingList)> = terms
            .into_iter()
            .map(|t| {
                let list = self.postings_for(&t).unwrap_or(&NO_POSTINGS);
                (t, list)
            })
            .collect();
        // Shortest first: the result can never outgrow the shortest list, so
        // starting there bounds every later intersection by the rarest term
        // instead of dragging the longest list through each step. The sort is
        // stable, so equally common terms stay in alphabetical order.
        lists.sort_by_key(|(_, l)| l.len());

        if let (Some(plan), Some(started)) = (plan.as_deref_mut(), started) {
            plan.terms = lists
                .iter()
                .map(|(t, l)| PlanTerm {
                    term: t.clone(),
                    df: l.len(),
                    idf: idf(self.num_docs(), l.len()),
                })
                .collect();
            plan.lookup_time = started.elapsed();
        }
        let Some(&(_, first)) = lists.first() else {
            return (Vec::new(), Vec::new());
        };

        let mut docs = first.doc_ids();
        for (term, list) in &lists[1..] {
            if docs.is_empty() {
                break;
            }
            let start = plan.is_some().then(Instant::now);
            let next = list.intersect(&docs, strategy);
            if let (Some(plan), Some(start)) = (plan.as_deref_mut(), start) {
                plan.steps.push(PlanStep {
                    term: term.clone(),
                    candidates: docs.len(),
                    df: list.len(),
                    algorithm: strategy.resolve(docs.len(), list.len()),
                    matched: next.len(),
                    time: start.elapsed(),
                });
            }
            docs = next;
        }
        (lists.into_iter().map(|(_, l)| l).collect(), docs)
    }

    /// BM25 score for each of `docs`, which contain every term in `lists`:
    ///
    ///   sum over terms of  idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
    ///
    /// tf saturates (k1 caps what repetition adds), and a doc longer than
    /// average needs more occurrences for the same score (b).
    fn score(&self, lists: &[&PostingList], docs: &[DocId]) -> Vec<f32> {
        let avg_len = self.avg_doc_len() as f32;
        // The length part of the denominator depends only on the doc, so
        // compute it once per doc rather than once per (doc, term).
        let norms: Vec<f32> = docs
            .iter()
            .map(|&d| {
                let len = self.doc_lens[d as usize] as f32;
                BM25_K1 * (1.0 - BM25_B + BM25_B * len / avg_len)
            })
            .collect();

        let mut scores = vec![0.0f32; docs.len()];
        let mut tfs = Vec::with_capacity(docs.len());
        for list in lists {
            let idf = idf(self.num_docs(), list.len());
            list.tfs_for(docs, &mut tfs);
            for ((score, &tf), &norm) in scores.iter_mut().zip(&tfs).zip(&norms) {
                let tf = tf as f32;
                *score += idf * tf * (BM25_K1 + 1.0) / (tf + norm);
            }
        }
        scores
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

    fn index(docs: &[&str]) -> Index {
        let mut idx = Index::new();
        for (i, text) in docs.iter().enumerate() {
            idx.add_document(format!("d{i}"), text);
        }
        idx
    }

    fn ranked_ids(idx: &Index, query: &str) -> Vec<DocId> {
        idx.search_ranked(query, 10).hits.iter().map(|h| h.doc).collect()
    }

    #[test]
    fn term_frequencies_are_counted_per_doc() {
        let idx = sample();
        let list = idx.postings_for("learning").unwrap();
        assert_eq!(list.len(), 3); // df counts documents, not occurrences
        let mut tfs = Vec::new();
        list.tfs_for(&[0, 2, 3], &mut tfs);
        assert_eq!(tfs, vec![1, 2, 1]); // doc c says "learning" twice
    }

    #[test]
    fn bm25_matches_hand_computed_score() {
        // N = 2, df(rust) = 1, avg_len = 1.5, doc 0 has len 1 and tf 1.
        // idf  = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln 2
        // norm = 1.2 * (1 - 0.75 + 0.75 * 1 / 1.5) = 0.9
        // score = ln 2 * 1 * 2.2 / (1 + 0.9)
        let idx = index(&["rust", "go go"]);
        let hits = idx.search_ranked("rust", 10).hits;
        let expected = 2f32.ln() * 2.2 / 1.9;
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - expected).abs() < 1e-6, "{} vs {expected}", hits[0].score);
    }

    #[test]
    fn bm25_rewards_term_frequency_and_short_docs() {
        // More occurrences in docs of equal length ranks higher...
        let idx = index(&["rust go go", "rust rust go"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
        // ...and the same count in a shorter doc ranks higher.
        let idx = index(&["rust a b c d e", "rust a"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
    }

    #[test]
    fn bm25_weights_rare_terms_more() {
        // "machine" is in 3 docs, "rust" in 2. Docs 0 and 1 both match the
        // query with the same length; doc 1 has more of the rarer term.
        let idx = index(&["machine machine rust", "machine rust rust", "machine"]);
        assert_eq!(ranked_ids(&idx, "machine rust"), vec![1, 0]);
    }

    #[test]
    fn ranking_is_top_k_with_deterministic_ties() {
        let idx = index(&["rust", "rust", "rust rust", "rust"]);
        let r = idx.search_ranked("rust", 2);
        assert_eq!(r.matched, 4);
        // Doc 2 scores highest; docs 0, 1 and 3 tie, and the lowest id wins.
        assert_eq!(r.hits.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![2, 0]);
        assert!(idx.search_ranked("rust", 0).hits.is_empty());
    }

    #[test]
    fn repeated_query_terms_count_once() {
        let idx = sample();
        let once = idx.search_ranked("rust", 10).hits;
        let twice = idx.search_ranked("rust RUST rust", 10).hits;
        assert_eq!(once, twice);
        assert_eq!(once[0].score, twice[0].score);
    }

    #[test]
    fn explain_reports_order_steps_and_same_results_as_search() {
        let idx = sample();
        let plan = idx.explain("machine learning rust", 10);
        // rust (df 2) first; learning and machine (df 3) tie, alphabetical.
        let order: Vec<&str> = plan.terms.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(order, vec!["rust", "learning", "machine"]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!((plan.steps[0].candidates, plan.steps[0].df), (2, 3));
        assert_eq!(plan.steps[0].algorithm, Strategy::Merge); // 1.5x < 16x
        assert_eq!(plan.hits, idx.search_ranked("machine learning rust", 10).hits);
        assert_eq!(plan.matched, 1);
    }

    #[test]
    fn explain_picks_gallop_for_skewed_lists() {
        let mut docs = vec!["common rare"];
        docs.extend(std::iter::repeat_n("common", 40));
        let plan = index(&docs).explain("common rare", 10);
        assert_eq!(plan.steps[0].algorithm, Strategy::Gallop); // 41 vs 1
        assert!(plan.to_string().contains("gallop"));
    }

    #[test]
    fn explain_stops_at_an_unknown_term() {
        let plan = sample().explain("machine zebra learning", 10);
        assert_eq!(plan.terms[0].term, "zebra");
        assert_eq!(plan.terms[0].df, 0);
        assert!(plan.steps.is_empty()); // nothing left to intersect
        assert_eq!(plan.matched, 0);
        assert!(plan.to_string().contains("never read"));
    }

    #[test]
    fn unknown_term_or_empty_query_matches_nothing() {
        let idx = sample();
        assert!(idx.search("machine zebra").is_empty());
        assert!(idx.search("").is_empty());
        assert!(idx.search("!!!").is_empty());
    }
}

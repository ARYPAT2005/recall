//! One index: term dictionary, posting lists and the query path.

use crate::hash::FxBuildHasher;
use crate::postings::{PostingList, NO_POSTINGS};
use crate::rank::{idf, top_k, Hit, Ranked, BM25_B, BM25_K1};
use crate::tokenize::{for_each_token, tokenize};
use crate::{DocId, Plan, PlanStep, PlanTerm, Strategy};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

/// An inverted index over a set of documents. A shard is one of these.
#[derive(Default)]
pub struct Index {
    // word -> term id
    term_ids: HashMap<String, u32, FxBuildHasher>,
    // term id -> posting list
    lists: Vec<PostingList>,
    // scratch space for add_document
    pending_tf: Vec<u32>,
    touched: Vec<u32>,
    pub doc_names: Vec<String>,
    // words per document, for BM25
    pub doc_lens: Vec<u32>,
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

    pub fn num_postings(&self) -> usize {
        self.lists.iter().map(|l| l.len()).sum()
    }

    /// (bytes used, bytes allocated) across all posting lists.
    pub fn posting_bytes(&self) -> (usize, usize) {
        self.lists
            .iter()
            .map(|l| l.heap_bytes())
            .fold((0, 0), |(u, a), (lu, la)| (u + lu, a + la))
    }

    pub fn avg_doc_len(&self) -> f64 {
        if self.doc_lens.is_empty() {
            return 0.0;
        }
        self.total_len as f64 / self.doc_lens.len() as f64
    }

    pub fn terms(&self) -> impl Iterator<Item = (&str, &PostingList)> {
        self.term_ids.iter().map(|(t, &id)| (t.as_str(), &self.lists[id as usize]))
    }

    /// Indexes one document and returns its id.
    pub fn add_document(&mut self, name: String, text: &str) -> DocId {
        let id = self.doc_names.len() as DocId;
        let Index { term_ids, lists, pending_tf, touched, .. } = self;
        let mut len = 0u32;

        // Count each word first, then append one entry per word, since encoded lists can't be edited.
        for_each_token(text, &mut String::new(), |word| {
            len += 1;
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

    pub fn postings_for(&self, term: &str) -> Option<&PostingList> {
        self.term_ids.get(term).map(|&id| &self.lists[id as usize])
    }

    /// Doc ids that contain every word in the query.
    pub fn search(&self, query: &str) -> Vec<DocId> {
        self.search_with(query, Strategy::Adaptive)
    }

    /// `search` with a fixed intersection strategy, for benchmarking.
    pub fn search_with(&self, query: &str, strategy: Strategy) -> Vec<DocId> {
        self.matching(&self.weigh(query), strategy, None).1
    }

    /// The k best matches, ranked by BM25.
    pub fn search_ranked(&self, query: &str, k: usize) -> Ranked {
        self.rank_terms(&self.weigh(query), self.avg_doc_len() as f32, k)
    }

    /// Runs a query and records every step. Runs it twice and keeps the warm run.
    pub fn explain(&self, query: &str, k: usize) -> Plan {
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
        let started = Instant::now();
        let terms = self.weigh(query);
        plan.lookup_time = started.elapsed();
        let (lists, docs) = self.matching(&terms, Strategy::Adaptive, Some(&mut plan));
        let start = Instant::now();
        let ranked = self.rank(&terms, &lists, &docs, self.avg_doc_len() as f32, k);
        plan.rank_time = start.elapsed();
        plan.matched = ranked.matched;
        plan.hits = ranked.hits;
        plan
    }

    fn weigh(&self, query: &str) -> Vec<QueryTerm> {
        weigh(query, self.num_docs(), |t| self.postings_for(t).map_or(0, |l| l.len()))
    }

    /// BM25 top k using the stats in `terms`. Shards get corpus-wide stats here.
    pub(crate) fn rank_terms(&self, terms: &[QueryTerm], avg_len: f32, k: usize) -> Ranked {
        let (lists, docs) = self.matching(terms, Strategy::Adaptive, None);
        self.rank(terms, &lists, &docs, avg_len, k)
    }

    fn rank(
        &self,
        terms: &[QueryTerm],
        lists: &[&PostingList],
        docs: &[DocId],
        avg_len: f32,
        k: usize,
    ) -> Ranked {
        let scores = self.score(terms, lists, docs, avg_len);
        let hits = docs.iter().zip(&scores).map(|(&doc, &score)| Hit { doc, score });
        Ranked {
            matched: docs.len(),
            hits: top_k(hits, k),
        }
    }

    /// Each term's posting list, and the docs that are in all of them.
    fn matching(
        &self,
        terms: &[QueryTerm],
        strategy: Strategy,
        mut plan: Option<&mut Plan>,
    ) -> (Vec<&PostingList>, Vec<DocId>) {
        let started = plan.is_some().then(Instant::now);
        // Unknown words get an empty list, so the AND comes out empty.
        let lists: Vec<&PostingList> = terms
            .iter()
            .map(|t| self.postings_for(&t.term).unwrap_or(&NO_POSTINGS))
            .collect();

        // Rarest first: the result can never be bigger than the shortest list.
        let mut order: Vec<usize> = (0..lists.len()).collect();
        order.sort_by_key(|&i| lists[i].len());

        if let (Some(plan), Some(started)) = (plan.as_deref_mut(), started) {
            plan.terms = order
                .iter()
                .map(|&i| PlanTerm {
                    term: terms[i].term.clone(),
                    df: lists[i].len(),
                    idf: terms[i].idf,
                })
                .collect();
            plan.lookup_time += started.elapsed();
        }
        let Some(&first) = order.first() else {
            return (lists, Vec::new());
        };

        let mut docs = lists[first].doc_ids();
        for &i in &order[1..] {
            if docs.is_empty() {
                break;
            }
            let list = lists[i];
            let start = plan.is_some().then(Instant::now);
            let next = list.intersect(&docs, strategy);
            if let (Some(plan), Some(start)) = (plan.as_deref_mut(), start) {
                plan.steps.push(PlanStep {
                    term: terms[i].term.clone(),
                    candidates: docs.len(),
                    df: list.len(),
                    algorithm: strategy.resolve(docs.len(), list.len()),
                    matched: next.len(),
                    time: start.elapsed(),
                });
            }
            docs = next;
        }
        (lists, docs)
    }

    /// BM25 score for each doc. Terms are added in a fixed order so shards match a single index exactly.
    fn score(
        &self,
        terms: &[QueryTerm],
        lists: &[&PostingList],
        docs: &[DocId],
        avg_len: f32,
    ) -> Vec<f32> {
        let norms: Vec<f32> = docs
            .iter()
            .map(|&d| {
                let len = self.doc_lens[d as usize] as f32;
                BM25_K1 * (1.0 - BM25_B + BM25_B * len / avg_len)
            })
            .collect();

        let mut scores = vec![0.0f32; docs.len()];
        let mut tfs = Vec::with_capacity(docs.len());
        for (term, list) in terms.iter().zip(lists) {
            list.tfs_for(docs, &mut tfs);
            for ((score, &tf), &norm) in scores.iter_mut().zip(&tfs).zip(&norms) {
                let tf = tf as f32;
                *score += term.idf * tf * (BM25_K1 + 1.0) / (tf + norm);
            }
        }
        scores
    }

    /// Indexes every .txt file in a directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> io::Result<Index> {
        let mut idx = Index::new();
        let mut paths: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect();
        paths.sort();

        for path in paths {
            let text = fs::read_to_string(&path)?;
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            idx.add_document(name, &text);
        }
        Ok(idx)
    }

    /// Indexes a corpus file with one `title<TAB>body` document per line.
    pub fn from_corpus_file<P: AsRef<Path>>(path: P) -> io::Result<Index> {
        Index::from_corpus_range(path.as_ref(), 0, u64::MAX)
    }

    /// Same, but only the lines in bytes [start, end).
    pub fn from_corpus_range(path: &Path, start: u64, end: u64) -> io::Result<Index> {
        let mut idx = Index::new();
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(start))?;
        let reader = BufReader::with_capacity(1 << 20, file.take(end.saturating_sub(start)));

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

    pub(crate) fn total_len(&self) -> u64 {
        self.total_len
    }
}

/// A query word with its document frequency and idf.
pub(crate) struct QueryTerm {
    pub term: String,
    pub df: usize,
    pub idf: f32,
}

/// Tokenizes and dedupes a query, orders the words by df and works out each idf.
pub(crate) fn weigh(query: &str, num_docs: usize, df: impl Fn(&str) -> usize) -> Vec<QueryTerm> {
    let mut terms = tokenize(query);
    terms.sort_unstable();
    terms.dedup();
    let mut with_df: Vec<(usize, String)> = terms.into_iter().map(|t| (df(&t), t)).collect();
    with_df.sort_by_key(|&(df, _)| df);
    with_df
        .into_iter()
        .map(|(df, term)| QueryTerm {
            idf: idf(num_docs, df),
            df,
            term,
        })
        .collect()
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
        assert_eq!(list.len(), 3);
        let mut tfs = Vec::new();
        list.tfs_for(&[0, 2, 3], &mut tfs);
        assert_eq!(tfs, vec![1, 2, 1]);
    }

    #[test]
    fn bm25_matches_hand_computed_score() {
        let idx = index(&["rust", "go go"]);
        let hits = idx.search_ranked("rust", 10).hits;
        let expected = 2f32.ln() * 2.2 / 1.9;
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - expected).abs() < 1e-6, "{} vs {expected}", hits[0].score);
    }

    #[test]
    fn bm25_rewards_term_frequency_and_short_docs() {
        let idx = index(&["rust go go", "rust rust go"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
        let idx = index(&["rust a b c d e", "rust a"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
    }

    #[test]
    fn bm25_weights_rare_terms_more() {
        let idx = index(&["machine machine rust", "machine rust rust", "machine"]);
        assert_eq!(ranked_ids(&idx, "machine rust"), vec![1, 0]);
    }

    #[test]
    fn ranking_is_top_k_with_deterministic_ties() {
        let idx = index(&["rust", "rust", "rust rust", "rust"]);
        let r = idx.search_ranked("rust", 2);
        assert_eq!(r.matched, 4);
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
        let order: Vec<&str> = plan.terms.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(order, vec!["rust", "learning", "machine"]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!((plan.steps[0].candidates, plan.steps[0].df), (2, 3));
        assert_eq!(plan.steps[0].algorithm, Strategy::Merge);
        assert_eq!(plan.hits, idx.search_ranked("machine learning rust", 10).hits);
        assert_eq!(plan.matched, 1);
    }

    #[test]
    fn explain_picks_gallop_for_skewed_lists() {
        let mut docs = vec!["common rare"];
        docs.extend(std::iter::repeat_n("common", 40));
        let plan = index(&docs).explain("common rare", 10);
        assert_eq!(plan.steps[0].algorithm, Strategy::Gallop);
        assert!(plan.to_string().contains("gallop"));
    }

    #[test]
    fn explain_stops_at_an_unknown_term() {
        let plan = sample().explain("machine zebra learning", 10);
        assert_eq!(plan.terms[0].term, "zebra");
        assert_eq!(plan.terms[0].df, 0);
        assert!(plan.steps.is_empty());
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

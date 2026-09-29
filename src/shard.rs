//! Document-partitioned shards: the corpus is split into contiguous ranges of
//! documents, each indexed by its own `Index` on its own thread.
//!
//! Partitioning by document, not by term, means every shard can answer every
//! query on its own - an AND never needs posting lists from two shards. The
//! price is that every query visits every shard, and BM25 needs statistics
//! (document count, average length, each term's df) that no single shard
//! has. Scoring with a shard's local statistics would make scores from
//! different shards incomparable, so the coordinator computes corpus-wide
//! statistics and hands them to every shard.

use crate::index::{weigh, QueryTerm};
use crate::rank::{top_k, Hit, Ranked};
use crate::{DocId, Index, BLOCK};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::thread;

/// How a query reaches the shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanOut {
    /// One shard after another on the calling thread.
    Sequential,
    /// All shards at once, on the shards' worker threads.
    Parallel,
    /// Parallel only when the query is expensive enough to pay for threads.
    Adaptive,
}

impl FanOut {
    pub const ALL: [FanOut; 3] = [FanOut::Sequential, FanOut::Parallel, FanOut::Adaptive];
}

/// Adaptive fans out in parallel once a query's estimated work reaches this
/// many scored-entry units (see `estimated_work`). A single-term query's
/// work is just its df, and the df sweep in bench.rs puts the break-even
/// between ~1,000 (parallel 1.00x) and ~2,000 (1.33x) with the worker pool.
pub const PARALLEL_MIN_WORK: f64 = 1_500.0;

/// Decoding a posting entry costs about a fifth of scoring one: ~2 ns per
/// entry for a merge over a decoded list (V7 ratio sweep) against ~10 ns per
/// entry for a large single-term query (V8 df sweep).
const DECODE_VS_SCORE: f64 = 5.0;

/// How much work a query will be, in units of one posting entry scored,
/// using only corpus-wide document frequencies - so it's known before any
/// shard is touched. `dfs` are in execution order, rarest first.
///
/// The rarest list is decoded and its entries become the candidates. Each
/// later list is either merged (decoding all of it) or galloped (decoding
/// about one block per candidate), whichever decodes less. Then, assuming
/// terms occur independently - the classic query-optimizer assumption - a
/// fraction df / N of the candidates survive each step.
fn estimated_work(dfs: &[usize], num_docs: usize) -> f64 {
    let Some((&rarest, rest)) = dfs.split_first() else {
        return 0.0;
    };
    let mut candidates = rarest as f64;
    let mut decoded = 0.0;
    for &df in rest {
        decoded += (df as f64).min(candidates * BLOCK as f64);
        candidates *= df as f64 / num_docs.max(1) as f64;
    }
    rarest as f64 + decoded / DECODE_VS_SCORE
}

pub struct ShardedIndex {
    /// Shared with the worker threads, which is why it's behind an Arc.
    shards: Arc<Vec<Index>>,
    /// bases[s] is the global id of shard s's first document. Shards hold
    /// contiguous id ranges in corpus order, so global id = base + local id,
    /// and global ids come out the same as a single index would assign.
    bases: Vec<DocId>,
    num_docs: usize,
    total_len: u64,
    /// workers[i] serves shard i + 1. Shard 0 runs on the calling thread,
    /// which would otherwise sit idle waiting for the others.
    workers: Vec<Worker>,
}

/// A long-lived thread that answers queries for one shard. Starting a thread
/// per query cost ~45 µs of overhead (see BENCHMARKS.md, V8), more than most
/// queries take; a parked worker only has to be woken up.
struct Worker {
    jobs: Option<mpsc::Sender<Job>>,
    handle: Option<thread::JoinHandle<()>>,
}

/// One query for one shard: the weighted terms (shared by all shards, so one
/// allocation per query), and a channel for the answer.
struct Job {
    terms: Arc<Vec<QueryTerm>>,
    avg_len: f32,
    k: usize,
    reply: mpsc::Sender<(usize, Ranked)>,
}

impl Worker {
    fn spawn(shards: Arc<Vec<Index>>, shard: usize) -> Worker {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let handle = thread::Builder::new()
            .name(format!("recall-shard-{shard}"))
            .spawn(move || {
                // Ends when the ShardedIndex drops its sender.
                for job in inbox {
                    let ranked = shards[shard].rank_terms(&job.terms, job.avg_len, job.k);
                    // The caller only goes away if it panicked; nothing to do.
                    let _ = job.reply.send((shard, ranked));
                }
            })
            .expect("failed to start shard worker thread");
        Worker {
            jobs: Some(jobs),
            handle: Some(handle),
        }
    }
}

impl Drop for ShardedIndex {
    fn drop(&mut self) {
        // Close every inbox first so all workers exit, then wait for them.
        for w in &mut self.workers {
            w.jobs.take();
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl ShardedIndex {
    /// Wrap already-built shards, and start a worker thread for each shard
    /// after the first. Their documents, in shard order, are the corpus in
    /// order.
    pub fn from_shards(shards: Vec<Index>) -> Self {
        let mut bases = Vec::with_capacity(shards.len());
        let mut num_docs = 0usize;
        for shard in &shards {
            bases.push(num_docs as DocId);
            num_docs += shard.num_docs();
        }
        let total_len = shards.iter().map(|s| s.total_len()).sum();
        let shards = Arc::new(shards);
        let workers = (1..shards.len())
            .map(|s| Worker::spawn(Arc::clone(&shards), s))
            .collect();
        ShardedIndex {
            shards,
            bases,
            num_docs,
            total_len,
            workers,
        }
    }

    /// Index a corpus file (one `title<TAB>body` document per line) into
    /// `num_shards` shards, building them in parallel, one thread each.
    ///
    /// The file is cut into byte ranges at line boundaries, and each thread
    /// opens the file and reads only its own range. Nothing is shared between
    /// threads while indexing, so there are no locks: each thread owns the
    /// Index it builds and hands it back when it finishes.
    pub fn from_corpus_file<P: AsRef<Path>>(path: P, num_shards: usize) -> io::Result<Self> {
        let path = path.as_ref();
        let bounds = split_at_lines(path, num_shards.max(1))?;
        let shards = thread::scope(|s| {
            let handles: Vec<_> = bounds
                .windows(2)
                .map(|w| {
                    let (start, end) = (w[0], w[1]);
                    s.spawn(move || Index::from_corpus_range(path, start, end))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("indexing thread panicked"))
                .collect::<io::Result<Vec<Index>>>()
        })?;
        Ok(ShardedIndex::from_shards(shards))
    }

    pub fn shards(&self) -> &[Index] {
        &self.shards
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avg_doc_len(&self) -> f64 {
        if self.num_docs == 0 {
            return 0.0;
        }
        self.total_len as f64 / self.num_docs as f64
    }

    /// Corpus-wide document frequency: the sum of every shard's. Exact,
    /// because each document lives in exactly one shard.
    pub fn df(&self, term: &str) -> usize {
        self.shards
            .iter()
            .map(|s| s.postings_for(term).map_or(0, |l| l.len()))
            .sum()
    }

    /// Distinct terms across all shards. A common term is in every shard's
    /// dictionary, so this has to deduplicate: O(total dictionary size).
    pub fn num_terms(&self) -> usize {
        let mut seen = HashSet::new();
        for shard in self.shards.iter() {
            seen.extend(shard.terms().map(|(t, _)| t));
        }
        seen.len()
    }

    pub fn num_postings(&self) -> usize {
        self.shards.iter().map(|s| s.num_postings()).sum()
    }

    /// Posting-list heap bytes summed over shards: (used, allocated).
    pub fn posting_bytes(&self) -> (usize, usize) {
        self.shards
            .iter()
            .map(|s| s.posting_bytes())
            .fold((0, 0), |(u, a), (su, sa)| (u + su, a + sa))
    }

    /// The name of the document with this global id.
    pub fn doc_name(&self, id: DocId) -> &str {
        // The last shard starting at or before id. An empty shard has the
        // same base as the one after it, and "last" skips past it.
        let s = self.bases.partition_point(|&b| b <= id) - 1;
        &self.shards[s].doc_names[(id - self.bases[s]) as usize]
    }

    /// The k most relevant documents containing every query term, by BM25,
    /// with results identical to one Index over the whole corpus.
    pub fn search_ranked(&self, query: &str, k: usize) -> Ranked {
        self.search_ranked_with(query, k, FanOut::Adaptive)
    }

    pub fn search_ranked_with(&self, query: &str, k: usize, fan_out: FanOut) -> Ranked {
        // 1. Weigh terms with corpus-wide statistics, so every shard scores
        //    on the same scale. This is the step a naive sharded engine skips.
        let terms = weigh(query, self.num_docs, |t| self.df(t));
        let avg_len = self.avg_doc_len() as f32;

        // 2. Ask every shard for its own top k - in parallel only if the
        //    query is big enough to pay for handing work to other threads.
        let parallel = self.shards.len() > 1
            && match fan_out {
                FanOut::Sequential => false,
                FanOut::Parallel => true,
                FanOut::Adaptive => self.worth_parallel(&terms),
            };
        let per_shard = if parallel {
            self.each_shard_parallel(terms, avg_len, k)
        } else {
            self.shards
                .iter()
                .map(|s| s.rank_terms(&terms, avg_len, k))
                .collect()
        };

        // 3. Merge. The global top k must be within the union of the shards'
        //    top k: a document outside its own shard's top k has k better
        //    documents in that shard alone.
        let matched = per_shard.iter().map(|r| r.matched).sum();
        let candidates: Vec<Hit> = per_shard
            .into_iter()
            .zip(&self.bases)
            .flat_map(|(r, &base)| {
                r.hits.into_iter().map(move |h| Hit {
                    doc: base + h.doc,
                    score: h.score,
                })
            })
            .collect();
        Ranked {
            matched,
            hits: top_k(candidates, k),
        }
    }

    fn worth_parallel(&self, terms: &[QueryTerm]) -> bool {
        let dfs: Vec<usize> = terms.iter().map(|t| t.df).collect();
        estimated_work(&dfs, self.num_docs) >= PARALLEL_MIN_WORK
    }

    /// Run the query on every shard at once: each worker takes its shard,
    /// and the calling thread takes shard 0 while it waits.
    fn each_shard_parallel(&self, terms: Vec<QueryTerm>, avg_len: f32, k: usize) -> Vec<Ranked> {
        let terms = Arc::new(terms);
        let (reply, replies) = mpsc::channel();
        for w in &self.workers {
            let job = Job {
                terms: Arc::clone(&terms),
                avg_len,
                k,
                reply: reply.clone(),
            };
            w.jobs.as_ref().unwrap().send(job).expect("shard worker exited");
        }
        drop(reply); // so `replies` ends if a worker dies instead of hanging

        let mut out: Vec<Option<Ranked>> = (0..self.shards.len()).map(|_| None).collect();
        out[0] = Some(self.shards[0].rank_terms(&terms, avg_len, k));
        for (shard, ranked) in replies.iter().take(self.workers.len()) {
            out[shard] = Some(ranked);
        }
        out.into_iter()
            .map(|r| r.expect("shard worker panicked"))
            .collect()
    }
}

/// Byte offsets that cut a file into `n` ranges of roughly equal size, each
/// starting at the beginning of a line: [0, b1, ..., len]. A range can come
/// out empty if a single line is longer than a whole range.
fn split_at_lines(path: &Path, n: usize) -> io::Result<Vec<u64>> {
    let len = fs::metadata(path)?.len();
    let mut reader = BufReader::new(File::open(path)?);
    let mut bounds = vec![0];
    let mut rest_of_line = Vec::new();
    for i in 1..n {
        let target = len * i as u64 / n as u64;
        let prev = *bounds.last().unwrap();
        let bound = if target <= prev {
            prev
        } else {
            // Start one byte early: if target is already a line start, that
            // byte is the '\n' ending the line before, and we stop right there.
            reader.seek(SeekFrom::Start(target - 1))?;
            rest_of_line.clear();
            target - 1 + reader.read_until(b'\n', &mut rest_of_line)? as u64
        };
        bounds.push(bound);
    }
    bounds.push(len);
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small Zipf-ish corpus: short words, some very common, most rare.
    fn corpus(docs: usize, seed: u64) -> Vec<String> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..docs)
            .map(|_| {
                let len = 1 + next() % 30;
                (0..len)
                    .map(|_| {
                        // Squaring a uniform number skews toward small ids.
                        let r = (next() % 1000) as f64 / 1000.0;
                        format!("w{}", (r * r * 400.0) as u32)
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    fn single(docs: &[String]) -> Index {
        let mut idx = Index::new();
        for (i, d) in docs.iter().enumerate() {
            idx.add_document(format!("d{i}"), d);
        }
        idx
    }

    /// Split docs into contiguous shards at the given cut points.
    fn sharded(docs: &[String], cuts: &[usize]) -> ShardedIndex {
        let mut bounds = vec![0];
        bounds.extend_from_slice(cuts);
        bounds.push(docs.len());
        let shards = bounds
            .windows(2)
            .map(|w| {
                let mut idx = Index::new();
                for (i, doc) in docs.iter().enumerate().take(w[1]).skip(w[0]) {
                    idx.add_document(format!("d{i}"), doc);
                }
                idx
            })
            .collect();
        ShardedIndex::from_shards(shards)
    }

    #[test]
    fn sharded_results_equal_single_index_bit_for_bit() {
        let docs = corpus(2_000, 42);
        let one = single(&docs);
        let queries: Vec<String> = (0..300)
            .map(|i| match i % 3 {
                0 => format!("w{}", i % 50),
                1 => format!("w{} w{}", i % 7, i % 97),
                _ => format!("w{} w{} w{}", i % 3, i % 11, i % 200),
            })
            .collect();
        // Uneven cuts and an empty shard (1000..1000) on purpose.
        for cuts in [vec![], vec![1000], vec![300, 1000, 1000, 1700], vec![1, 2, 3, 1999]] {
            let many = sharded(&docs, &cuts);
            for q in &queries {
                let want = one.search_ranked(q, 10);
                for fan_out in FanOut::ALL {
                    let got = many.search_ranked_with(q, 10, fan_out);
                    assert_eq!(got.matched, want.matched, "{q:?} {cuts:?}");
                    // Hit equality compares scores exactly, not approximately.
                    assert_eq!(got.hits, want.hits, "{q:?} {cuts:?} {fan_out:?}");
                    for (g, w) in got.hits.iter().zip(&want.hits) {
                        assert_eq!(g.score.to_bits(), w.score.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn scores_use_corpus_wide_statistics() {
        // Shard 0 is full of "b", shard 1 full of "a". By local statistics
        // "a" is rare in shard 0 and "b" rare in shard 1, so each shard would
        // inflate its own match. Corpus-wide the two terms are equally
        // common, docs 0 and 12 score the same, and the lower id wins.
        let mut docs = vec!["a a b".to_string()];
        docs.extend(std::iter::repeat_n("b".to_string(), 10));
        docs.push("x".to_string());
        docs.push("a b b".to_string());
        docs.extend(std::iter::repeat_n("a".to_string(), 10));
        let one = single(&docs);
        let many = sharded(&docs, &[12]);
        let want = one.search_ranked("a b", 10);
        assert_eq!(many.search_ranked("a b", 10).hits, want.hits);
        assert_eq!(want.hits[0].doc, 0);
        assert_eq!(want.hits[0].score, want.hits[1].score);
    }

    #[test]
    fn many_threads_can_query_one_sharded_index() {
        // Queries from several threads share the same workers; each query
        // carries its own reply channel, so answers can't get crossed.
        let docs = corpus(1_000, 11);
        let one = single(&docs);
        let many = sharded(&docs, &[250, 500, 750]);
        thread::scope(|s| {
            for t in 0..4 {
                let (one, many) = (&one, &many);
                s.spawn(move || {
                    for i in 0..200 {
                        let q = format!("w{} w{}", (i + t) % 9, (i * 7 + t) % 150);
                        let got = many.search_ranked_with(&q, 5, FanOut::Parallel);
                        assert_eq!(got.hits, one.search_ranked(&q, 5).hits);
                    }
                });
            }
        });
    }

    #[test]
    fn work_estimate_separates_cheap_and_expensive_queries() {
        let n = 100_000;
        // Single term: the work is its df.
        assert_eq!(estimated_work(&[1_000], n), 1_000.0);
        // Rare + common: gallops, decoding ~one block per rare candidate.
        assert!(estimated_work(&[50, 30_000], n) < PARALLEL_MIN_WORK);
        // Mid + common: few candidates, but each costs a block of a huge list.
        assert!(estimated_work(&[500, 30_000], n) >= PARALLEL_MIN_WORK);
        // Common + common: many candidates to score.
        assert!(estimated_work(&[20_000, 50_000], n) >= PARALLEL_MIN_WORK);
        // Candidates shrink after each step, so the third list costs little.
        let two = estimated_work(&[50, 20_000], n);
        let three = estimated_work(&[50, 20_000, 50_000], n);
        assert!(three - two < 150.0, "{two} -> {three}");
    }

    #[test]
    fn doc_names_map_global_ids_across_empty_shards() {
        let docs = corpus(10, 7);
        let many = sharded(&docs, &[0, 4, 4, 9]);
        assert_eq!(many.num_docs(), 10);
        for i in 0..10 {
            assert_eq!(many.doc_name(i as DocId), format!("d{i}"));
        }
    }

    #[test]
    fn parallel_build_from_file_matches_single_index() {
        let docs = corpus(500, 3);
        let mut text = String::new();
        for (i, d) in docs.iter().enumerate() {
            text.push_str(&format!("d{i}\t{d}\n"));
            if i == 250 {
                text.push_str("a line with no tab is skipped\n");
            }
        }
        text.pop(); // no trailing newline on the last line
        let path = std::env::temp_dir().join(format!("recall-shard-test-{}.tsv", std::process::id()));
        fs::write(&path, &text).unwrap();

        let one = Index::from_corpus_file(&path).unwrap();
        for n in [1, 2, 3, 7, 64] {
            let bounds = split_at_lines(&path, n).unwrap();
            let bytes = text.as_bytes();
            for &b in &bounds[1..bounds.len() - 1] {
                assert!(b == 0 || b as usize == bytes.len() || bytes[b as usize - 1] == b'\n');
            }
            let many = ShardedIndex::from_corpus_file(&path, n).unwrap();
            assert_eq!(many.num_shards(), n);
            assert_eq!(many.num_docs(), one.num_docs());
            assert_eq!(many.num_postings(), one.num_postings());
            for i in 0..one.num_docs() {
                assert_eq!(many.doc_name(i as DocId), one.doc_names[i]);
            }
            for q in ["w0", "w1 w2", "w3 w30 w300", "nope"] {
                assert_eq!(many.search_ranked(q, 10).hits, one.search_ranked(q, 10).hits);
            }
        }
        fs::remove_file(&path).unwrap();
    }
}

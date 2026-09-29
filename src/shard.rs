//! Splits the index into shards that are built and searched on parallel threads.

use crate::index::{weigh, QueryTerm};
use crate::rank::{top_k, Hit, Ranked};
use crate::{DocId, Index, BLOCK};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::thread;

/// How a query gets sent to the shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanOut {
    /// One shard at a time, on the calling thread.
    Sequential,
    /// All shards at once, on the worker threads.
    Parallel,
    /// Parallel only when the query is big enough to be worth it.
    Adaptive,
}

impl FanOut {
    pub const ALL: [FanOut; 3] = [FanOut::Sequential, FanOut::Parallel, FanOut::Adaptive];
}

/// Estimated work above which queries run in parallel (from the sweep in bench.rs).
pub const PARALLEL_MIN_WORK: f64 = 1_500.0;

/// Decoding a posting entry costs about 1/5 as much as scoring one.
const DECODE_VS_SCORE: f64 = 5.0;

/// Estimates a query's work from corpus-wide dfs, assuming words occur independently.
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

/// Shards over contiguous ranges of documents, with a worker thread per extra shard.
pub struct ShardedIndex {
    // shared with the worker threads
    shards: Arc<Vec<Index>>,
    // global id of each shard's first document
    bases: Vec<DocId>,
    num_docs: usize,
    total_len: u64,
    // workers[i] serves shard i + 1; the calling thread handles shard 0
    workers: Vec<Worker>,
}

/// A long-running thread that answers queries for one shard.
struct Worker {
    jobs: Option<mpsc::Sender<Job>>,
    handle: Option<thread::JoinHandle<()>>,
}

/// One query for one shard, plus a channel for the answer.
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
                // Runs until the ShardedIndex drops its end of the channel.
                for job in inbox {
                    let ranked = shards[shard].rank_terms(&job.terms, job.avg_len, job.k);
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
        // Close the channels so the workers exit, then wait for them.
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
    /// Wraps built shards and starts their worker threads.
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

    /// Builds `num_shards` shards from a corpus file in parallel, one thread each.
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

    /// A word's document frequency across all shards.
    pub fn df(&self, term: &str) -> usize {
        self.shards
            .iter()
            .map(|s| s.postings_for(term).map_or(0, |l| l.len()))
            .sum()
    }

    /// Distinct words across all shards.
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

    pub fn posting_bytes(&self) -> (usize, usize) {
        self.shards
            .iter()
            .map(|s| s.posting_bytes())
            .fold((0, 0), |(u, a), (su, sa)| (u + su, a + sa))
    }

    /// Name of the document with this global id.
    pub fn doc_name(&self, id: DocId) -> &str {
        let s = self.bases.partition_point(|&b| b <= id) - 1;
        &self.shards[s].doc_names[(id - self.bases[s]) as usize]
    }

    /// The k best matches by BM25. Same results as one unsharded index.
    pub fn search_ranked(&self, query: &str, k: usize) -> Ranked {
        self.search_ranked_with(query, k, FanOut::Adaptive)
    }

    pub fn search_ranked_with(&self, query: &str, k: usize, fan_out: FanOut) -> Ranked {
        // Use corpus-wide stats so every shard scores the same way.
        let terms = weigh(query, self.num_docs, |t| self.df(t));
        let avg_len = self.avg_doc_len() as f32;

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

        // The overall top k is always somewhere in the shards' top k lists.
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

    /// Sends the query to every worker and handles shard 0 on this thread.
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
        drop(reply);

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

/// Byte offsets that split a file into n parts, each starting at a line.
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
            // Start one byte early in case target is already a line start.
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
        for cuts in [vec![], vec![1000], vec![300, 1000, 1000, 1700], vec![1, 2, 3, 1999]] {
            let many = sharded(&docs, &cuts);
            for q in &queries {
                let want = one.search_ranked(q, 10);
                for fan_out in FanOut::ALL {
                    let got = many.search_ranked_with(q, 10, fan_out);
                    assert_eq!(got.matched, want.matched, "{q:?} {cuts:?}");
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
        assert_eq!(estimated_work(&[1_000], n), 1_000.0);
        assert!(estimated_work(&[50, 30_000], n) < PARALLEL_MIN_WORK);
        assert!(estimated_work(&[500, 30_000], n) >= PARALLEL_MIN_WORK);
        assert!(estimated_work(&[20_000, 50_000], n) >= PARALLEL_MIN_WORK);
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
        text.pop();
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

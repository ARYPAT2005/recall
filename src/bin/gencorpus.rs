//! Generates a synthetic corpus so we have something big enough to measure.
//!
//! Usage: cargo run --release --bin gencorpus -- <num_docs> <out_path>
//!
//! Two design choices worth understanding:
//!
//! 1. DETERMINISTIC. A hand-rolled xorshift PRNG with a fixed seed means the
//!    same corpus every run. Benchmarks comparing version A to version B are
//!    meaningless if the input changed underneath you.
//!
//! 2. ZIPFIAN word distribution. Real text is wildly skewed: "the" appears in
//!    every document, "photosynthesis" in a handful. If we picked words
//!    uniformly, every posting list would be the same length and both BM25's
//!    IDF term and the intersection optimization would look pointless. Zipf's
//!    law (frequency proportional to 1/rank) reproduces the real skew.

use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// xorshift64*: tiny, fast, good enough for generating test data.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform float in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

fn load_vocab(size: usize) -> Vec<String> {
    let raw = std::fs::read_to_string("/usr/share/dict/words").unwrap_or_default();
    let mut words: Vec<String> = raw
        .lines()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| w.len() >= 3 && w.len() <= 12 && w.chars().all(|c| c.is_ascii_alphabetic()))
        .collect();
    words.dedup();

    if words.len() < size {
        // Fallback if the system dictionary is missing.
        for i in words.len()..size {
            words.push(format!("term{i}"));
        }
    }
    words.truncate(size);
    words
}

/// Cumulative distribution for Zipf: P(rank i) proportional to 1/(i+1).
fn zipf_cdf(n: usize) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for i in 0..n {
        acc += 1.0 / (i as f64 + 1.0);
        cdf.push(acc);
    }
    let total = acc;
    for v in cdf.iter_mut() {
        *v /= total; // normalize to [0, 1]
    }
    cdf
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let num_docs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let out_path = args.get(2).map(|s| s.as_str()).unwrap_or("corpus/docs.tsv");

    const VOCAB_SIZE: usize = 50_000;
    let vocab = load_vocab(VOCAB_SIZE);
    let cdf = zipf_cdf(vocab.len());
    let mut rng = Rng(0x9E3779B97F4A7C15); // fixed seed

    let mut out = BufWriter::with_capacity(1 << 20, File::create(out_path)?);
    let mut total_tokens = 0usize;

    for id in 0..num_docs {
        let len = rng.next_range(60, 200); // words per document
        write!(out, "doc{id:07}.txt\t")?;
        for i in 0..len {
            // Inverse-transform sampling: pick a uniform value, binary search
            // the CDF to find which rank it lands in.
            let u = rng.next_f64();
            let rank = cdf.partition_point(|&c| c < u).min(vocab.len() - 1);
            if i > 0 {
                out.write_all(b" ")?;
            }
            out.write_all(vocab[rank].as_bytes())?;
        }
        out.write_all(b"\n")?;
        total_tokens += len;
    }
    out.flush()?;

    println!(
        "Wrote {num_docs} documents ({total_tokens} tokens, vocab {}) to {out_path}",
        vocab.len()
    );
    Ok(())
}

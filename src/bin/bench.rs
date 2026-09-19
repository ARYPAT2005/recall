//! The benchmark harness. This is the tool that turns the project into resume
//! bullets, so it gets built BEFORE the optimizations, not after.
//!
//! Usage: cargo run --release --bin bench -- [corpus_path]
//!
//! Reports: indexing throughput, query latency percentiles, peak memory.

use recall::{tokenize, Index};
use std::env;
use std::time::{Duration, Instant};

/// Peak resident set size in bytes, straight from the kernel via getrusage(2).
/// macOS reports ru_maxrss in bytes; Linux reports kilobytes.
fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0;
        }
        let maxrss = usage.ru_maxrss as u64;
        if cfg!(target_os = "macos") {
            maxrss
        } else {
            maxrss * 1024
        }
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.1} {}", UNITS[u])
}

/// Percentile from an already-sorted slice, using nearest-rank.
fn pct(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn main() -> std::io::Result<()> {
    if cfg!(debug_assertions) {
        eprintln!("!! Running in DEBUG. Numbers are meaningless. Use --release.\n");
    }

    let corpus = env::args().nth(1).unwrap_or_else(|| "corpus/docs.tsv".into());

    // ---------- Indexing ----------
    let t0 = Instant::now();
    let index = Index::from_corpus_file(&corpus)?;
    let index_time = t0.elapsed();

    let docs = index.num_docs();
    let postings = index.num_postings();

    println!("== Indexing ==");
    println!("  corpus          {corpus}");
    println!("  documents       {docs}");
    println!("  unique terms    {}", index.num_terms());
    println!("  posting entries {postings}");
    println!("  avg doc length  {:.1} tokens", index.avg_doc_len());
    println!("  wall time       {:.3} s", index_time.as_secs_f64());
    println!(
        "  throughput      {:.0} docs/sec",
        docs as f64 / index_time.as_secs_f64()
    );

    // ---------- Query latency ----------
    // Draw query terms from the index itself, so we measure realistic lookups
    // rather than a pile of misses. Sample across the frequency spectrum:
    // common terms (long posting lists) and rare ones (short) behave very
    // differently, and an average over only rare terms would flatter us.
    let mut terms: Vec<(&String, usize)> =
        index.postings.iter().map(|(t, p)| (t, p.len())).collect();
    terms.sort_by_key(|&(_, len)| std::cmp::Reverse(len));

    let sample: Vec<&str> = terms
        .iter()
        .step_by((terms.len() / 500).max(1))
        .map(|&(t, _)| t.as_str())
        .collect();

    // Warm up: first touches pull the hash table into cache. Measuring those
    // would conflate cache misses with the algorithm's real cost.
    for t in sample.iter().take(100) {
        std::hint::black_box(index.postings_for(t));
    }

    const ITERS: usize = 20_000;
    let mut latencies = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let term = sample[i % sample.len()];
        let t = Instant::now();
        // black_box stops the optimizer from deleting work whose result we
        // never use - a classic way to accidentally benchmark nothing.
        std::hint::black_box(index.postings_for(std::hint::black_box(term)));
        latencies.push(t.elapsed());
    }
    latencies.sort_unstable();

    println!("\n== Single-term query latency ({ITERS} queries) ==");
    println!("  P50   {:>9.3} us", pct(&latencies, 50.0).as_secs_f64() * 1e6);
    println!("  P95   {:>9.3} us", pct(&latencies, 95.0).as_secs_f64() * 1e6);
    println!("  P99   {:>9.3} us", pct(&latencies, 99.0).as_secs_f64() * 1e6);
    println!("  max   {:>9.3} us", latencies.last().unwrap().as_secs_f64() * 1e6);

    // ---------- Memory ----------
    println!("\n== Memory ==");
    println!("  peak RSS        {}", human_bytes(peak_rss_bytes()));
    println!(
        "  bytes/posting   {:.1}",
        peak_rss_bytes() as f64 / postings.max(1) as f64
    );

    std::hint::black_box(&index);
    let _ = tokenize("keep the tokenizer linked");
    Ok(())
}

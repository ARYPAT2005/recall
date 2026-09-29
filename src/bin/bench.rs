//! The benchmark harness. This is the tool that turns the project into resume
//! bullets, so it gets built BEFORE the optimizations, not after.
//!
//! Usage: cargo run --release --bin bench -- [corpus_path] [--shards N] [--index-only]
//!
//! Reports: indexing throughput, query latency percentiles, peak memory.
//! With --shards N > 1, builds N shards in parallel and measures fan-out.
//! --index-only times the build 5 times and stops: parallel builds are
//! sensitive to background load, so one run isn't enough.

use recall::{FanOut, Index, PostingList, ShardedIndex, Strategy};
use std::collections::HashMap;
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

/// Runs one query, returns how many documents matched.
type QueryFn<'a> = dyn Fn(&str) -> usize + 'a;

/// Average nanoseconds per call. Runs in batches long enough (~2 ms) that
/// timer resolution doesn't matter, and takes the fastest batch to shed noise
/// from interrupts and frequency scaling.
fn time_ns<R>(mut f: impl FnMut() -> R) -> f64 {
    let t = Instant::now();
    std::hint::black_box(f());
    let once = t.elapsed().as_nanos().max(1) as usize;
    let reps = (2_000_000 / once).clamp(1, 100_000);
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(f());
        }
        best = best.min(t.elapsed().as_nanos() as f64 / reps as f64);
    }
    best
}

/// Percentile from an already-sorted slice, using nearest-rank.
fn pct(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

const Q_ITERS: usize = 5_000;

/// Terms with document frequencies, most frequent first. Ties are broken by
/// term: HashMap iteration order is randomly seeded, so sorting on df alone
/// would draw different queries every run.
fn by_df(mut terms: Vec<(&str, usize)>) -> Vec<(&str, usize)> {
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    terms
}

/// Query classes, 500 fixed queries each. Terms are bucketed by rank in the
/// frequency ordering; classes mix buckets because common+rare is where the
/// algorithms differ most.
fn query_classes(terms: &[(&str, usize)]) -> Vec<(&'static str, Vec<String>)> {
    let rank_range = |lo: f64, hi: f64| -> Vec<&str> {
        let n = terms.len() as f64;
        terms[(lo * n) as usize..(hi * n) as usize]
            .iter()
            .map(|&(t, _)| t)
            .collect()
    };
    let common = rank_range(0.0, 0.002); // top 100 terms
    let mid = rank_range(0.02, 0.1);
    let rare = rank_range(0.3, 0.6);

    let mut rng: u64 = 0x2545F4914F6CDD1D;
    let mut pick = |bucket: &[&str]| -> String {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        bucket[(rng % bucket.len() as u64) as usize].to_string()
    };
    vec![
        ("common + common", (0..500).map(|_| format!("{} {}", pick(&common), pick(&common))).collect()),
        ("common + mid", (0..500).map(|_| format!("{} {}", pick(&common), pick(&mid))).collect()),
        ("common + rare", (0..500).map(|_| format!("{} {}", pick(&common), pick(&rare))).collect()),
        ("mid + mid", (0..500).map(|_| format!("{} {}", pick(&mid), pick(&mid))).collect()),
        ("rare + rare", (0..500).map(|_| format!("{} {}", pick(&rare), pick(&rare))).collect()),
        ("common x2 + rare", (0..500).map(|_| format!("{} {} {}", pick(&common), pick(&common), pick(&rare))).collect()),
    ]
}

/// Times `run` over the class's queries; `run` returns the hit count.
fn measure(queries: &[String], run: &QueryFn) -> (Vec<Duration>, f64) {
    for q in queries.iter().take(100) {
        std::hint::black_box(run(q));
    }
    let mut lat = Vec::with_capacity(Q_ITERS);
    let mut hits = 0usize;
    for i in 0..Q_ITERS {
        let q = &queries[i % queries.len()];
        let t = Instant::now();
        hits += std::hint::black_box(run(std::hint::black_box(q)));
        lat.push(t.elapsed());
    }
    lat.sort_unstable();
    (lat, hits as f64 / Q_ITERS as f64)
}

fn print_latency_header() {
    println!(
        "  {:<18} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "class", "mode", "P50 us", "P95 us", "P99 us", "avg matched"
    );
}

fn print_latency_row(class: &str, mode: &str, lat: &[Duration], avg: f64) {
    println!(
        "  {:<18} {:>11} {:>11.3} {:>11.3} {:>11.3} {:>11.1}",
        class,
        mode,
        pct(lat, 50.0).as_secs_f64() * 1e6,
        pct(lat, 95.0).as_secs_f64() * 1e6,
        pct(lat, 99.0).as_secs_f64() * 1e6,
        avg
    );
}

fn print_indexing(corpus: &str, docs: usize, terms: usize, postings: usize, avg_len: f64, time: Duration) {
    println!("  corpus          {corpus}");
    println!("  documents       {docs}");
    println!("  unique terms    {terms}");
    println!("  posting entries {postings}");
    println!("  avg doc length  {avg_len:.1} tokens");
    println!("  wall time       {:.3} s", time.as_secs_f64());
    println!("  throughput      {:.0} docs/sec", docs as f64 / time.as_secs_f64());
}

fn print_memory((used, allocated): (usize, usize), postings: usize) {
    println!("\n== Memory ==");
    println!("  peak RSS        {}", human_bytes(peak_rss_bytes()));
    // "used" is what the entries occupy; "allocated" adds Vec growth space.
    for (label, bytes) in [("used", used), ("allocated", allocated)] {
        println!(
            "  postings {label:<9} {} ({:.2} bytes/posting)",
            human_bytes(bytes as u64),
            bytes as f64 / postings.max(1) as f64
        );
    }
    println!(
        "  RSS/posting     {:.1} bytes",
        peak_rss_bytes() as f64 / postings.max(1) as f64
    );
}

fn main() -> std::io::Result<()> {
    if cfg!(debug_assertions) {
        eprintln!("!! Running in DEBUG. Numbers are meaningless. Use --release.\n");
    }

    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut shards = 1;
    if let Some(i) = args.iter().position(|a| a == "--shards") {
        shards = args.get(i + 1).and_then(|n| n.parse().ok()).expect("--shards N");
        args.drain(i..i + 2);
    }
    let index_only = args.iter().any(|a| a == "--index-only");
    args.retain(|a| a != "--index-only");
    let corpus = args.first().cloned().unwrap_or_else(|| "corpus/docs.tsv".into());

    if index_only {
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            if shards > 1 {
                std::hint::black_box(ShardedIndex::from_corpus_file(&corpus, shards)?);
            } else {
                std::hint::black_box(Index::from_corpus_file(&corpus)?);
            }
            times.push(t.elapsed().as_secs_f64());
        }
        let shown: Vec<String> = times.iter().map(|t| format!("{t:.3}")).collect();
        times.sort_by(f64::total_cmp);
        println!("{shards} shard(s): median {:.3} s  (runs: {})", times[2], shown.join(" "));
        return Ok(());
    }

    if shards > 1 {
        run_sharded(&corpus, shards)
    } else {
        run_single(&corpus)
    }
}

fn run_single(corpus: &str) -> std::io::Result<()> {
    // ---------- Indexing ----------
    let t0 = Instant::now();
    let index = Index::from_corpus_file(corpus)?;
    let index_time = t0.elapsed();
    let postings = index.num_postings();

    println!("== Indexing ==");
    print_indexing(corpus, index.num_docs(), index.num_terms(), postings, index.avg_doc_len(), index_time);

    // ---------- Query latency ----------
    // Draw query terms from the index itself, so we measure realistic lookups
    // rather than a pile of misses. Sample across the frequency spectrum:
    // common terms (long posting lists) and rare ones (short) behave very
    // differently, and an average over only rare terms would flatter us.
    let terms = by_df(index.terms().map(|(t, l)| (t, l.len())).collect());

    let sample: Vec<&str> = terms
        .iter()
        .step_by((terms.len() / 500).max(1))
        .map(|&(t, _)| t)
        .collect();

    // A lookup takes less than one tick of the clock (Apple Silicon's timer
    // runs at 24 MHz, so one tick = 41.67 ns). Timing lookups one at a time
    // just reports the tick - V1's "41 ns P50" was exactly that. So time a
    // pass over all sampled terms and divide; no per-call percentiles here.
    // black_box stops the optimizer from deleting work whose result we never
    // use - a classic way to accidentally benchmark nothing.
    let lookup_ns = time_ns(|| {
        for t in &sample {
            std::hint::black_box(index.postings_for(std::hint::black_box(t)));
        }
    }) / sample.len() as f64;

    println!("\n== Single-term lookup ({} terms, batched) ==", sample.len());
    println!("  avg   {lookup_ns:>9.1} ns");

    // ---------- Intersection: merge vs gallop by length ratio ----------
    // Pair one long list with shorter lists at controlled length ratios and
    // time both algorithms on the exact same pair. The crossover ratio is
    // where Adaptive should switch (GALLOP_RATIO in lib.rs).
    let by_len: Vec<&PostingList> =
        terms.iter().map(|&(t, _)| index.postings_for(t).unwrap()).collect();
    // Never pair a list with itself: every comparison would be Equal, the
    // branch perfectly predicted, and the ratio-1 row meaninglessly fast.
    let closest = |long: &PostingList, target: usize| -> &PostingList {
        by_len
            .iter()
            .filter(|l| !std::ptr::eq(**l, long))
            .min_by_key(|l| l.len().abs_diff(target))
            .copied()
            .unwrap()
    };
    // Several long lists, so one unlucky term doesn't decide the answer.
    let longs: Vec<&PostingList> = [0usize, 10, 50, 200].iter().map(|&r| by_len[r]).collect();

    println!("\n== Intersection by length ratio (ns per intersection, avg of long lists) ==");
    println!("  {:>6}  {:>10}  {:>10}  {:>8}", "ratio", "merge", "gallop", "speedup");
    for ratio in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let (mut merge_ns, mut gallop_ns) = (0.0, 0.0);
        for &long in &longs {
            // Materialize the short side outside the timer: in a real query
            // it's the running result, already a plain sorted Vec.
            let short = closest(long, long.len() / ratio).doc_ids();
            merge_ns += time_ns(|| long.intersect(&short, Strategy::Merge));
            gallop_ns += time_ns(|| long.intersect(&short, Strategy::Gallop));
        }
        let n = longs.len() as f64;
        println!(
            "  {:>6}  {:>10.0}  {:>10.0}  {:>7.2}x",
            ratio,
            merge_ns / n,
            gallop_ns / n,
            merge_ns / gallop_ns
        );
    }

    // ---------- Multi-term query latency ----------
    let classes = query_classes(&terms);
    println!("\n== Multi-term query latency ({Q_ITERS} queries/class, end-to-end search) ==");
    println!("  Boolean rows return every match by id. BM25 rows score every match;");
    println!("  top-10 keeps the best 10 in a heap; 'all' sorts every match.");
    print_latency_header();
    for (name, queries) in &classes {
        let modes: [(&str, &QueryFn); 5] = [
            ("Merge", &|q| index.search_with(q, Strategy::Merge).len()),
            ("Gallop", &|q| index.search_with(q, Strategy::Gallop).len()),
            ("Adaptive", &|q| index.search_with(q, Strategy::Adaptive).len()),
            ("BM25 top-10", &|q| index.search_ranked(q, 10).matched),
            ("BM25 all", &|q| index.search_ranked(q, usize::MAX).matched),
        ];
        for (mode, run) in modes {
            let (lat, avg) = measure(queries, run);
            print_latency_row(name, mode, &lat, avg);
        }
    }

    print_memory(index.posting_bytes(), postings);
    std::hint::black_box(&index);
    Ok(())
}

fn run_sharded(corpus: &str, n: usize) -> std::io::Result<()> {
    // ---------- Indexing: one thread per shard ----------
    let t0 = Instant::now();
    let index = ShardedIndex::from_corpus_file(corpus, n)?;
    let index_time = t0.elapsed();
    let postings = index.num_postings();

    println!("== Indexing ({n} shards, one thread each) ==");
    print_indexing(corpus, index.num_docs(), index.num_terms(), postings, index.avg_doc_len(), index_time);
    let sizes: Vec<usize> = index.shards().iter().map(|s| s.num_docs()).collect();
    println!("  docs per shard  {sizes:?}");

    // Corpus-wide df per term: each shard only knows its own share.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for shard in index.shards() {
        for (t, l) in shard.terms() {
            *df.entry(t).or_default() += l.len();
        }
    }
    let terms = by_df(df.into_iter().collect());

    // ---------- Fan-out: sequential vs parallel by df ----------
    // A single-term query's cost grows with its df: every matching doc gets
    // decoded and scored. Parallel fan-out splits that work but pays to hand
    // it to the worker threads, so it should only win above some df. Sweep df
    // to find where (PARALLEL_MIN_WORK in shard.rs).
    println!("\n== Fan-out by df (single-term BM25 top-10, us per query, 20 terms each) ==");
    println!("  {:>8}  {:>12}  {:>12}  {:>8}", "df", "sequential", "parallel", "speedup");
    for target in [30usize, 100, 300, 1_000, 2_000, 3_000, 10_000, 30_000, 100_000] {
        // The 20 terms whose df is closest to the target.
        let mut near: Vec<&(&str, usize)> = terms.iter().collect();
        near.sort_by_key(|&&(t, d)| (d.abs_diff(target), t));
        let queries: Vec<&str> = near.iter().take(20).map(|&&(t, _)| t).collect();
        let avg_df = near.iter().take(20).map(|&&(_, d)| d).sum::<usize>() / 20;
        let per_query = |fan_out: FanOut| {
            time_ns(|| {
                for q in &queries {
                    std::hint::black_box(index.search_ranked_with(q, 10, fan_out));
                }
            }) / queries.len() as f64
                / 1e3
        };
        let (seq, par) = (per_query(FanOut::Sequential), per_query(FanOut::Parallel));
        println!("  {avg_df:>8}  {seq:>12.2}  {par:>12.2}  {:>7.2}x", seq / par);
    }

    // ---------- Multi-term query latency ----------
    let classes = query_classes(&terms);
    println!("\n== Multi-term BM25 top-10 latency ({Q_ITERS} queries/class, {n} shards) ==");
    print_latency_header();
    for (name, queries) in &classes {
        for fan_out in FanOut::ALL {
            let (lat, avg) = measure(queries, &|q| index.search_ranked_with(q, 10, fan_out).matched);
            print_latency_row(name, &format!("{fan_out:?}"), &lat, avg);
        }
    }

    print_memory(index.posting_bytes(), postings);
    std::hint::black_box(&index);
    Ok(())
}

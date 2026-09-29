//! Interactive search REPL.
//!
//! Usage:
//!   cargo run --release                           -> indexes ./documents/*.txt
//!   cargo run --release -- corpus/docs.tsv        -> indexes a corpus file
//!   cargo run --release -- corpus/docs.tsv --shards 4
//!                                                 -> 4 shards, built in parallel
//!   cargo run --release -- [path] explain <query> -> print one query's plan
//!
//! In the REPL, `:explain <query>` prints the plan instead of just results.
//! --shards applies to .tsv corpora; a directory is always one shard.

use recall::{Index, Ranked, ShardedIndex};
use std::io::{self, Write};

fn load(path: Option<&str>, shards: usize) -> io::Result<ShardedIndex> {
    match path {
        Some(p) if p.ends_with(".tsv") => ShardedIndex::from_corpus_file(p, shards),
        Some(p) => Ok(ShardedIndex::from_shards(vec![Index::from_dir(p)?])),
        None => Ok(ShardedIndex::from_shards(vec![Index::from_dir("documents")?])),
    }
}

fn print_results(index: &ShardedIndex, results: &Ranked) {
    if results.matched == 0 {
        println!("No results.");
        return;
    }
    println!("{} result(s), best first by BM25:", results.matched);
    for hit in &results.hits {
        println!("  {:>7.3}  {}", hit.score, index.doc_name(hit.doc));
    }
    if results.matched > results.hits.len() {
        println!("  ... and {} more", results.matched - results.hits.len());
    }
}

fn explain(index: &ShardedIndex, query: &str) {
    // A plan describes one index's execution. With one shard that index is
    // the whole corpus, so its plan and statistics are exactly the query's.
    let [shard] = index.shards() else {
        println!("explain describes a single index; run without --shards to use it.");
        return;
    };
    let plan = shard.explain(query, 10);
    println!("{plan}");
    let results = Ranked {
        matched: plan.matched,
        hits: plan.hits,
    };
    print_results(index, &results);
}

fn main() -> io::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut shards = 1;
    if let Some(i) = args.iter().position(|a| a == "--shards") {
        shards = match args.get(i + 1).and_then(|n| n.parse().ok()) {
            Some(n) if n >= 1 => n,
            _ => {
                eprintln!("--shards needs a number >= 1");
                std::process::exit(2);
            }
        };
        args.drain(i..i + 2);
    }
    // `[path] explain <query...>`: everything after "explain" is the query.
    let (path, one_shot) = match args.iter().position(|a| a == "explain") {
        Some(i) => (args[..i].first(), Some(args[i + 1..].join(" "))),
        None => (args.first(), None),
    };
    let index = load(path.map(String::as_str), shards)?;

    println!(
        "Indexed {} documents into {} shard(s): {} unique terms, {} posting entries.",
        index.num_docs(),
        index.num_shards(),
        index.num_terms(),
        index.num_postings()
    );

    if let Some(query) = one_shot {
        println!();
        explain(&index, &query);
        return Ok(());
    }

    loop {
        print!("\nSearch: ");
        io::stdout().flush()?;

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break; // Ctrl-D
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match line.strip_prefix(":explain") {
            Some(query) => explain(&index, query.trim()),
            None => print_results(&index, &index.search_ranked(line, 10)),
        }
    }
    Ok(())
}

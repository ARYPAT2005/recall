//! Interactive search REPL.
//!
//! Usage:
//!   cargo run --release                          -> indexes ./documents/*.txt
//!   cargo run --release -- corpus/docs.tsv       -> indexes a corpus file
//!   cargo run --release -- [path] explain <query> -> print one query's plan
//!
//! In the REPL, `:explain <query>` prints the plan instead of just results.

use recall::{Index, Ranked};
use std::io::{self, Write};

fn load(path: Option<&str>) -> io::Result<Index> {
    match path {
        Some(p) if p.ends_with(".tsv") => Index::from_corpus_file(p),
        Some(p) => Index::from_dir(p),
        None => Index::from_dir("documents"),
    }
}

fn print_results(index: &Index, results: &Ranked) {
    if results.matched == 0 {
        println!("No results.");
        return;
    }
    println!("{} result(s), best first by BM25:", results.matched);
    for hit in &results.hits {
        println!("  {:>7.3}  {}", hit.score, index.doc_names[hit.doc as usize]);
    }
    if results.matched > results.hits.len() {
        println!("  ... and {} more", results.matched - results.hits.len());
    }
}

fn explain(index: &Index, query: &str) {
    let plan = index.explain(query, 10);
    println!("{plan}");
    let results = Ranked {
        matched: plan.matched,
        hits: plan.hits,
    };
    print_results(index, &results);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `[path] explain <query...>`: everything after "explain" is the query.
    let (path, one_shot) = match args.iter().position(|a| a == "explain") {
        Some(i) => (args[..i].first(), Some(args[i + 1..].join(" "))),
        None => (args.first(), None),
    };
    let index = load(path.map(String::as_str))?;

    println!(
        "Indexed {} documents, {} unique terms, {} posting entries.",
        index.num_docs(),
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

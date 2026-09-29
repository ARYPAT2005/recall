//! Interactive search REPL.
//!
//! Usage:
//!   cargo run --release                   -> indexes ./documents/*.txt
//!   cargo run --release -- corpus/docs.tsv -> indexes a corpus file

use recall::Index;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let arg = std::env::args().nth(1);
    let index = match arg {
        Some(path) if path.ends_with(".tsv") => Index::from_corpus_file(&path)?,
        Some(path) => Index::from_dir(&path)?,
        None => Index::from_dir("documents")?,
    };

    println!(
        "Indexed {} documents, {} unique terms, {} posting entries.",
        index.num_docs(),
        index.num_terms(),
        index.num_postings()
    );

    loop {
        print!("\nSearch: ");
        io::stdout().flush()?;

        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break; // Ctrl-D
        }

        if line.trim().is_empty() {
            continue;
        }

        let results = index.search_ranked(&line, 10);
        if results.matched == 0 {
            println!("No results.");
        } else {
            println!("{} result(s), best first by BM25:", results.matched);
            for hit in &results.hits {
                println!("  {:>7.3}  {}", hit.score, index.doc_names[hit.doc as usize]);
            }
            if results.matched > results.hits.len() {
                println!("  ... and {} more", results.matched - results.hits.len());
            }
        }
    }
    Ok(())
}

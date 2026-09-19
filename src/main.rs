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

        let query = line.trim().to_lowercase();
        if query.is_empty() {
            continue;
        }

        let hits = index.postings_for(&query);
        if hits.is_empty() {
            println!("No results.");
        } else {
            println!("{} result(s):", hits.len());
            for id in hits.iter().take(10) {
                println!("  {}", index.doc_names[*id as usize]);
            }
            if hits.len() > 10 {
                println!("  ... and {} more", hits.len() - 10);
            }
        }
    }
    Ok(())
}

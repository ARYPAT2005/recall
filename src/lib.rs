//! Recall: a full-text search engine, built from scratch.
//!
//! This is the *library*. The binaries in `src/bin/` and `src/main.rs` all
//! import it. Keeping the engine in a lib (instead of one big main.rs) is what
//! makes sharding possible later: a shard is just this struct in its own
//! process.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// A document's identity is just its position in `doc_names`.
///
/// u32, not usize: 4 bytes instead of 8. Posting lists are the bulk of the
/// index, so this literally halves the memory of the largest structure in the
/// program. It caps us at ~4.3 billion documents, which is fine.
pub type DocId = u32;

/// "Machine Learning, fast!" -> ["machine", "learning", "fast"]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

#[derive(Default)]
pub struct Index {
    /// The inverted index: term -> ascending list of documents containing it.
    pub postings: HashMap<String, Vec<DocId>>,
    /// doc_names[id] -> the filename or title
    pub doc_names: Vec<String>,
    /// doc_lens[id] -> token count. Unused today; BM25 needs it in step 5.
    pub doc_lens: Vec<u32>,
}

impl Index {
    pub fn new() -> Self {
        Index::default()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_names.len()
    }

    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    /// Total entries across all posting lists - the real size driver.
    pub fn num_postings(&self) -> usize {
        self.postings.values().map(|v| v.len()).sum()
    }

    /// Average document length in tokens. BM25 needs this.
    pub fn avg_doc_len(&self) -> f64 {
        if self.doc_lens.is_empty() {
            return 0.0;
        }
        self.doc_lens.iter().map(|&l| l as f64).sum::<f64>() / self.doc_lens.len() as f64
    }

    pub fn add_document(&mut self, name: String, text: &str) -> DocId {
        let id = self.doc_names.len() as DocId;
        let tokens = tokenize(text);

        for word in tokens.iter() {
            let list = self.postings.entry(word.clone()).or_default();
            // Docs arrive in ascending id order, so a duplicate can only be the
            // final element. This single check keeps every list deduped AND
            // sorted - which is exactly what the intersection in step 4 needs.
            if list.last() != Some(&id) {
                list.push(id);
            }
        }

        self.doc_names.push(name);
        self.doc_lens.push(tokens.len() as u32);
        id
    }

    /// The posting list for one term. Empty slice if the term is unknown.
    pub fn postings_for(&self, term: &str) -> &[DocId] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Index every .txt file in a directory. Good for small hand-written corpora.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> io::Result<Index> {
        let mut idx = Index::new();
        let mut paths: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect();
        paths.sort(); // stable doc ids across runs

        for path in paths {
            let text = fs::read_to_string(&path)?;
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            idx.add_document(name, &text);
        }
        Ok(idx)
    }

    /// Index a corpus file: one document per line, `title<TAB>body`.
    ///
    /// 100K separate files would mean 100K `open`/`close` syscalls and a lot of
    /// wasted inodes. One file read sequentially is dramatically faster and is
    /// how real corpora ship.
    pub fn from_corpus_file<P: AsRef<Path>>(path: P) -> io::Result<Index> {
        let mut idx = Index::new();
        let reader = BufReader::with_capacity(1 << 20, File::open(path)?);

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
}

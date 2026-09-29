//! Recall: a full-text search engine, built from scratch.
//!
//! This is the *library*. The binaries in `src/bin/` and `src/main.rs` all
//! import it. Keeping the engine in a lib (instead of one big main.rs) is what
//! makes sharding possible: a shard is just an `Index` on its own thread.

mod explain;
mod hash;
mod index;
mod intersect;
mod postings;
mod rank;
mod tokenize;

pub use explain::{Plan, PlanStep, PlanTerm};
pub use hash::{FxBuildHasher, FxHasher};
pub use index::Index;
pub use intersect::{intersect_gallop, intersect_merge, intersect_with, Strategy, GALLOP_RATIO};
pub use postings::{PostingList, BLOCK};
pub use rank::{idf, top_k, Hit, Ranked, BM25_B, BM25_K1};
pub use tokenize::{for_each_token, tokenize};

/// A document's identity is just its position in `doc_names`.
///
/// u32, not usize: 4 bytes instead of 8. Posting lists are the bulk of the
/// index, so this literally halves the memory of the largest structure in the
/// program. It caps us at ~4.3 billion documents, which is fine.
pub type DocId = u32;

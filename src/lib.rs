//! Recall: a full-text search engine written from scratch.

mod explain;
mod hash;
mod index;
mod intersect;
mod postings;
mod rank;
mod shard;
mod tokenize;

pub use explain::{Plan, PlanStep, PlanTerm};
pub use hash::{FxBuildHasher, FxHasher};
pub use index::Index;
pub use intersect::{intersect_gallop, intersect_merge, intersect_with, Strategy, GALLOP_RATIO};
pub use postings::{PostingList, BLOCK};
pub use rank::{idf, top_k, Hit, Ranked, BM25_B, BM25_K1};
pub use shard::{FanOut, ShardedIndex, PARALLEL_MIN_WORK};
pub use tokenize::{for_each_token, tokenize};

/// Documents are numbered from 0. u32 instead of usize halves the posting lists.
pub type DocId = u32;

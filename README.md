# Recall

A full-text search engine written from scratch in Rust — inverted index, benchmark
harness, and a log of every optimization step with the numbers that justified it.

No search libraries, no async runtime, no dependencies except `libc` (used only to
read peak memory in the benchmark).

## Why

Search engines are a good excuse to care about memory layout. The posting lists are
the whole program: they dominate the footprint, they decide cache behavior, and every
interesting optimization is about making them smaller or making them intersect faster.
The rule for this project is that nothing gets optimized until it's measured, and every
version's numbers go in [BENCHMARKS.md](BENCHMARKS.md) before the next one starts.

## Current results

100,000 synthetic documents, Zipfian over a 50K vocabulary, ~129 tokens/doc,
9.9M posting entries. Apple Silicon, `--release`.

| Step | Version | Index time | Throughput | Query P50 | Peak RSS |
|------|---------|-----------|------------|-----------|----------|
| 3 | V1 baseline: single-thread, in-memory, single-term lookup | 1.348 s | 74,182 docs/s | 0.041 µs | 68.3 MB |

That 41 ns P50 is honest about what it measures: one `HashMap` lookup
returning a slice. No intersection, no scoring, no copying. It will get slower as real
query work lands, and watching it get slower is the point.

## Design notes

**`DocId` is `u32`, not `usize`.** Posting lists are the largest structure in the
program, so 4 bytes instead of 8 halves the memory of the thing that matters. It caps
the index at ~4.3 billion documents.

**Posting lists stay sorted and deduped for free.** Documents are indexed in ascending
id order, so a repeated term can only ever match the list's final element. One
`list.last()` check per token maintains both invariants — which is exactly the
precondition a merge-based intersection needs.

**The corpus generator is deterministic and Zipfian.** A fixed-seed xorshift64\* PRNG
means the same corpus every run, because A/B benchmarks are meaningless if the input
moves. The Zipfian distribution matters just as much: with uniformly sampled words
every posting list would be the same length, and both BM25's IDF term and the
intersection optimization would look pointless.

**The engine is a library, not a `main.rs`.** A shard is just this struct in its own
process, so keeping it importable is what makes sharding possible later.

**The benchmark was built before the optimizations**, not after. It samples query terms
across the frequency spectrum (common terms have long posting lists and behave nothing
like rare ones), warms the cache before timing, and wraps both argument and result in
`black_box` so the optimizer can't delete work whose result is never used.

## Running it

```bash
# Interactive REPL over the small hand-written corpus in documents/
cargo run --release

# Generate a 100K-document synthetic corpus (~104 MB, gitignored)
cargo run --release --bin gencorpus -- 100000 corpus/docs.tsv

# Benchmark: indexing throughput, query latency percentiles, peak RSS
cargo run --release --bin bench -- corpus/docs.tsv

# REPL over the generated corpus
cargo run --release -- corpus/docs.tsv
```

Queries are currently single-term — multi-term intersection is the next step.

## Layout

```
src/lib.rs           the engine: tokenizer, Index, posting lists, corpus loaders
src/main.rs          interactive search REPL
src/bin/gencorpus.rs deterministic Zipfian corpus generator
src/bin/bench.rs     benchmark harness (throughput, latency percentiles, peak RSS)
documents/           small hand-written corpus for sanity checks
BENCHMARKS.md        every version's numbers, in order
```

## Roadmap

- [x] V1: in-memory inverted index, single-term lookup, benchmark harness
- [ ] Multi-term boolean queries via sorted-list intersection
- [ ] BM25 relevance scoring (`doc_lens` and `avg_doc_len` are already tracked for it)
- [ ] Posting list compression (delta + varint)
- [ ] Parallel indexing
- [ ] Sharding across processes

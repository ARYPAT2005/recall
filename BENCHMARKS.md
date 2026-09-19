# Recall — benchmark log

Machine: Apple Silicon (aarch64-apple-darwin), Rust 1.98.1, `--release`.
Corpus: 100,000 synthetic docs, Zipfian over a 50K vocab, ~129 tokens/doc,
9.9M posting entries. Deterministic seed, so runs are comparable.

Record every version here **before** optimizing the next one.

| Step | Version | Index time | Throughput | Query P50 | Query P99 | Peak RSS |
|------|---------|-----------|------------|-----------|-----------|----------|
| 3 | V1 baseline: single-thread, in-memory, single-term lookup | 1.348 s | 74,182 docs/s | 0.041 µs | 0.042 µs | 68.3 MB |

## Notes

- **V1 baseline.** Query latency here is only a `HashMap` lookup returning a
  slice — no intersection, no scoring, no copying. 41 ns is roughly one hash
  plus one cache miss. It will get slower as real work is added; that is
  expected and is the thing worth measuring.
- 7.2 bytes/posting overall. The posting lists themselves are 4 bytes/entry
  (`u32`); the rest is `HashMap` overhead and the 50K `String` keys.

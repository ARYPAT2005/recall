# Recall — benchmark log

Machine: Apple M2 (aarch64-apple-darwin), Rust 1.98.1, `--release`.
Corpus: 100,000 synthetic docs, Zipfian over a 50K vocab, ~129 tokens/doc,
9.9M posting entries. Deterministic seed, so runs are comparable.

Record every version here **before** optimizing the next one.

Index time on this laptop varies ±15% run to run, so from V2 on it is the
median of 3 runs. Compare versions measured in the same session, not
against rows recorded on a different day.

## Indexing and memory

| Step | Version | Index time | Throughput | Single-term lookup P50 | Peak RSS |
|------|---------|-----------|------------|-----------|----------|
| 3 | V1 baseline: single-thread, in-memory, single-term lookup | 1.348 s | 74,182 docs/s | 0.041 µs | 68.3 MB |
| 4 | V2/V3 multi-term AND (indexing unchanged) | 1.567 s | 63,809 docs/s | 0.041 µs | 68.0 MB |

## Multi-term query latency

500 fixed queries per class, 5,000 timed searches each, end to end
(tokenize → look up → intersect). Terms are bucketed by frequency rank:
common = top 100 terms, mid = ranks 1K–5K, rare = ranks 15K–30K.
P50 / P99 in µs.

| Version | common+common | common+mid | common+rare | mid+mid | rare+rare | common×2+rare |
|---------|---------------|------------|-------------|---------|-----------|---------------|
| V2 linear merge | 132 / 387 | 21.2 / 84.2 | 16.0 / 78.6 | 2.88 / 6.63 | 0.63 / 1.00 | 36.2 / 133 |
| V3 adaptive merge/gallop | 133 / 370 | 9.46 / 24.5 | **2.33 / 4.13** | 2.88 / 6.54 | 0.58 / 0.96 | **2.96 / 8.04** |
| (galloping only, for reference) | 197 / 571 | 9.46 / 24.4 | 2.33 / 4.21 | 3.50 / 7.83 | 0.67 / 0.88 | 2.83 / 7.33 |

Average hits per query: 9,292 / 149 / 15.9 / 2.3 / 0.0 / 5.4.

### Merge vs gallop by length ratio (V3)

One long list intersected with a shorter one at a controlled length ratio,
averaged over 4 long lists. ns per intersection, fastest of 7 batches.

| Ratio | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 1024 |
|-------|---|---|---|---|----|----|----|-----|-----|------|
| Merge | 188,670 | 189,301 | 115,550 | 78,215 | 61,069 | 52,490 | 47,317 | 44,266 | 42,236 | 40,577 |
| Gallop | 320,161 | 233,709 | 153,843 | 83,569 | 44,474 | 22,538 | 12,692 | 7,474 | 4,889 | 1,252 |
| Gallop speedup | 0.59× | 0.81× | 0.75× | 0.94× | **1.37×** | 2.33× | 3.73× | 5.92× | 8.64× | 32.4× |

## Notes

- **V1 baseline.** Query latency here is only a `HashMap` lookup returning a
  slice — no intersection, no scoring, no copying. 41 ns is roughly one hash
  plus one cache miss. It will get slower as real work is added; that is
  expected and is the thing worth measuring.
- 7.2 bytes/posting overall. The posting lists themselves are 4 bytes/entry
  (`u32`); the rest is `HashMap` overhead and the 50K `String` keys.
- **V2 linear merge.** Terms are intersected rarest first, so the working
  set can never outgrow the rarest list. Merge is O(m + n): a common+rare
  query still walks the whole common list, which is why common+rare (16 µs)
  costs almost as much as common+mid.
- **V3 adaptive.** Galloping search probes the long list at +1, +2, +4… then
  binary-searches the bracket, O(m log(n/m)). The ratio sweep puts the
  crossover between 8× (gallop 0.94×) and 16× (gallop 1.37×), so `search`
  gallops when the longer list is ≥16× the shorter and merges otherwise.
  Common+rare gets 6.9× faster at P50 and 19× at P99; common×2+rare 12×.
  Galloping alone would cost 49% more on common+common (197 vs 132 µs) —
  the adaptive switch keeps that case on merge.
- The first ratio sweep paired a list with itself at ratio 1: every
  comparison was Equal, the branch perfectly predicted, and merge looked 4×
  faster than it really is. The sweep now excludes self-pairs.
- Earlier runs drew slightly different queries each time because terms with
  equal document frequency came out in `HashMap` order, which is randomly
  seeded. Ties are now broken by term, so every run uses the same queries.

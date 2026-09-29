# Recall — benchmark log

Machine: Apple M2 (aarch64-apple-darwin), Rust 1.98.1, `--release`.
Corpus: 100,000 synthetic docs, Zipfian over a 50K vocab, ~129 tokens/doc,
9.9M posting entries. Deterministic seed, so runs are comparable.

Record every version here **before** optimizing the next one.

Index time on this laptop varies ±15% run to run, so from V2 on it is the
median of 3 runs. Compare versions measured in the same session, not
against rows recorded on a different day.

## Indexing and memory

| Step | Version | Index time | Throughput | Term lookup | Posting lists | Peak RSS |
|------|---------|-----------|------------|-------------|---------------|----------|
| 3 | V1 baseline: single-thread, in-memory, single-term lookup | 1.348 s | 74,182 docs/s | 14.9 ns † | 54.4 MB ‡ | 68.3 MB |
| 4 | V2/V3 multi-term AND (indexing unchanged) | 1.567 s | 63,809 docs/s | 14.9 ns † | 54.4 MB ‡ | 68.0 MB |
| 5 | V4 indexing cleanup: zero-copy tokenizer, no per-token key clone, FxHash | **0.499 s** | **200,527 docs/s** | **10.5 ns** | 54.4 MB ‡ | 68.8 MB |
| 6 | V5 BM25: per-doc term frequencies stored alongside doc ids | 0.565 s | 177,125 docs/s | 10.4 ns | 108.9 MB | 129.2 MB |

† V1 originally recorded 0.041 µs here, which was the clock tick, not the
lookup (see V4 notes). 14.9 ns is the same SipHash map re-measured with the
batched method.

‡ Posting-list bytes (Vec capacity, including unused growth space) were
first measured in V5. V5 stores doc ids and term frequencies in two Vecs
pushed in lockstep, so their capacities are identical and the doc-id half —
exactly what V1–V4 stored — is 108.9 / 2 = 54.4 MB.

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
| V5 BM25, top-10 via heap | 281 / 1,315 | 16.0 / 50.2 | 3.25 / 7.71 | 3.00 / 7.25 | 0.58 / 0.83 | 3.29 / 11.0 |
| (V5 BM25, sort every match) | 360 / 2,606 | 16.2 / 57.9 | 3.08 / 7.17 | 3.04 / 7.17 | 0.63 / 0.83 | 3.38 / 11.1 |

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
- **V4 indexing cleanup — 3.1× faster indexing (1.567 s → 0.499 s, same
  session).** Measured first: of 1,316 ms, reading lines was ~25 ms (2%),
  tokenizing ~570 ms (43%), and `HashMap` insertion ~720 ms (55%), across
  12.9M tokens. Each fix measured alone (fastest of 3):

  | Change | Full index |
  |---|---|
  | before | 1,316 ms |
  | tokenizer yields `&str` slices, no `Vec<String>` | 850 ms |
  | `get_mut` before insert — allocate a key only for new terms | 550 ms |
  | FxHash instead of SipHash | 470 ms |

  Reading was left alone because it's 2% of the time.
- The zero-copy tokenizer path only applies to already-lowercase ASCII
  words, and the synthetic corpus is 100% lowercase — so this benchmark
  flatters it. On real text, capitalized words take the next path:
  lowercased into a reused buffer (still no allocation). Non-ASCII words
  still allocate via `str::to_lowercase`, which is correct for cases like
  Greek final sigma.
- **The V1 "41 ns" lookup was the clock, not the code.** Every per-call
  latency came out as a multiple of 41.67 ns — one tick of Apple Silicon's
  24 MHz timer. A single lookup is shorter than one tick, so the bench now
  times a pass over all sampled terms and divides. Real numbers: 14.9 ns
  with SipHash, 10.5 ns with FxHash. Per-call percentiles elsewhere are
  still quantized to 41.67 ns, which is ~7% of the smallest one
  (rare+rare, ~0.6 µs).
- Multi-term query latency is unchanged by V4 (within noise), as expected:
  the query path's cost is intersection, not tokenizing or lookup.
- **V5 BM25.** Results are now ranked:
  `idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · len / avg_len))` summed over
  query terms, with k1 = 1.2, b = 0.75 and Lucene's always-positive idf. It
  still ranks only documents that contain every term (conjunctive BM25).
- BM25 needs each term's count in each doc, which V1–V4 threw away (the
  dedupe check dropped repeats). Storing it as a second `u32` per posting
  doubled the posting lists: 54.4 → 108.9 MB, peak RSS 68.8 → 129.2 MB.
  That's 8 bytes of data per posting plus ~44% unused Vec capacity from
  doubling growth. Indexing got 13% slower (0.499 → 0.565 s). This is the
  cost V7's compression goes after.
- Scoring costs little when few docs match (common+rare 2.33 → 3.25 µs) and
  a lot when many do: common+common goes 130 → 281 µs at P50 and 367 →
  1,315 µs at P99, scoring ~9.3K matches per query. `tfs_for` gallops once
  per candidate even when candidates are dense in the list, where a linear
  scan would do; left for now because V7 rewrites that lookup anyway.
- **Top-K by heap.** A max-heap of size k whose top is the worst hit kept;
  most new hits lose one comparison against it. Measured separately against
  the alternatives (k = 10, random scores):

  | Hits | Heap | Full sort | `select_nth_unstable` + sort k |
  |---|---|---|---|
  | 150 | 1.8 µs | 2.2 µs | 0.5 µs |
  | 9,300 | 16.6 µs | 217 µs | 31.9 µs |
  | 50,000 | 77.9 µs | 1,407 µs | 146 µs |

  The heap wins by 13–18× over sorting once there are thousands of matches,
  and loses to `select_nth_unstable` only below a few hundred, by about
  1 µs, so it's not worth a hybrid. End to end, top-10 vs sorting every
  match on common+common: 281 vs 360 µs P50, 1,315 vs 2,606 µs P99.
- The first version of the "sort every match" row pushed everything through
  the heap — really a heap sort, slower than `sort_unstable`, which made
  top-K look better than it is. `top_k` now sorts directly when k covers
  every hit, and that's what the row measures.

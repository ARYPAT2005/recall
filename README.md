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
9.9M posting entries. Apple M2, `--release`. Full numbers, methodology and
the dead ends are in [BENCHMARKS.md](BENCHMARKS.md).

| Version | What changed | Result |
|---------|--------------|--------|
| V1 | In-memory inverted index, single-term lookup | 1.35 s to index, 68.3 MB peak RSS |
| V2 | Multi-term AND via linear merge, rarest term first | common+rare query: 16.0 µs P50 |
| V3 | Adaptive merge / galloping intersection | common+rare **6.9× faster** (2.33 µs), common×2+rare 12× |
| V4 | Zero-copy tokenizer, no per-token key clone, FxHash | indexing **3.1× faster** (1.57 → 0.50 s) |
| V5 | BM25 ranking, top-k by heap | top-10 is 13× faster than sorting 9K matches; posting lists doubled to hold tf |
| V6 | `explain` query plans | showed BM25 scoring costing 10× the intersection on dense queries |
| V7 | Blocked delta + varint posting lists | posting lists **3.2× smaller**, peak RSS 127 → 53 MB; queries 1.4–3× slower |
| V8 | 4 document shards: parallel build, global BM25 stats, worker-pool fan-out | indexing **2.5× faster**, heavy queries **3.0× faster**; light queries 10–60% slower, peak RSS +66% |

Two measurement bugs turned up along the way and are written up in
BENCHMARKS.md: V1's "41 ns lookup" was one tick of the 24 MHz clock (the
real number is 15 ns, 10 ns after FxHash), and an early ratio sweep
intersected a list with itself.

## Design notes

**`DocId` is `u32`, not `usize`.** Posting lists are the largest structure in the
program, so 4 bytes instead of 8 halves the memory of the thing that matters. It caps
the index at ~4.3 billion documents.

**Posting lists stay sorted and deduped for free.** Documents are indexed in ascending
id order, and each document appends at most one entry per term, which is exactly the
precondition intersection needs.

**Rarest term first, then the right algorithm per step.** The result of an AND can
never be larger than its rarest term's list, so intersection starts there. Each step
then picks linear merge (O(m + n)) when the lists are similar in length or galloping
search (O(m log(n/m))) when one is much longer. The threshold (16×) came from a sweep
of both algorithms across length ratios, not a guess, and it had to be re-measured
when compression changed what each algorithm costs.

**BM25 with a top-k heap.** Scores use the standard k1 = 1.2, b = 0.75 and Lucene's
always-positive idf. Only the best k survive, in a max-heap whose top is the worst
hit kept, so most candidates cost one comparison. It beats sorting every match by
13–18× once there are thousands of them.

**Compressed posting lists that can still be searched.** Doc ids are stored as gaps,
and gaps and term frequencies are varints, so most take one byte instead of four.
Delta encoding alone saves nothing; the varint is what shrinks it. Varints can't be
indexed, so entries are grouped in 64-entry blocks with a skip table, and galloping
runs over the skip table and decodes only the blocks it needs. Lists are encoded
while indexing, so an uncompressed copy never exists and peak RSS drops too.

**Sharding by document, with corpus-wide statistics.** `--shards N` splits the corpus
into N contiguous ranges, cut at line boundaries in the file, and builds one `Index`
per range on its own thread. No locks are needed because no thread touches another's
index. Every query goes to every shard, and the per-shard top-k lists are merged. The
subtle part is BM25: each shard only knows its own document counts, and scoring with
those makes scores from different shards incomparable. So the coordinator sums each
term's df across shards and hands every shard the corpus-wide idf and average length.
Scores are also summed in a fixed term order, so sharded results are bit-identical to
a single index. That's tested at up to 64 shards.

**Parallel only when it pays.** Handing a query to other threads has a fixed cost, so
cheap queries run on the shards one after another. Before touching any shard, the
coordinator estimates the query's work from global dfs, the way a database planner
does: candidates start at the rarest term's df and shrink by df/N per term, assuming
terms occur independently. Shards run in parallel only above a threshold taken from a
measured sweep. Each shard has a long-lived worker thread waiting on a channel;
starting fresh threads per query cost ~45 µs, more than most queries take.

**`explain` runs the real code path.** The query plan is recorded by the same function
that answers normal searches, with tracing switched on, so it can't drift from what
search actually does. With tracing off, that path never reads the clock.

**The corpus generator is deterministic and Zipfian.** A fixed-seed xorshift64\* PRNG
means the same corpus every run, because A/B benchmarks are meaningless if the input
moves. The Zipfian distribution matters just as much: with uniformly sampled words
every posting list would be the same length, and both BM25's IDF term and the
intersection optimization would look pointless.

**The engine is a library, not a `main.rs`.** A shard is just this struct in its own
process, so keeping it importable is what makes sharding possible later.

**The benchmark was built before the optimizations**, not after. It groups queries by
term frequency (common terms have long posting lists and behave nothing like rare
ones), draws the same queries every run, warms the cache before timing, times
sub-tick operations in batches, and wraps arguments and results in `black_box` so
the optimizer can't delete work whose result is never used.

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

# Show the query plan: term order, which algorithm each step used and why, timings
cargo run --release -- corpus/docs.tsv explain "aal clamshell aam abas"

# Build 4 shards in parallel and search across them
cargo run --release -- corpus/docs.tsv --shards 4

# Benchmark sharded indexing and fan-out; --index-only just times the build
cargo run --release --bin bench -- corpus/docs.tsv --shards 4
cargo run --release --bin bench -- corpus/docs.tsv --shards 4 --index-only
```

Queries are ANDed and ranked by BM25. In the REPL, `:explain <query>` prints the plan:

```
QUERY PLAN  "aal clamshell aam abas"  over 100,000 docs

Terms, rarest first (that's the execution order):
  "clamshell"  df        34   idf  7.97
  "abas"       df    18,144   idf  1.71
  "aam"        df    95,989   idf  0.04
  "aal"        df    99,972   idf  0.00

Steps:
  1. tokenize, look up and order 4 term(s), start from "clamshell": 34 candidates   0.67 µs
  2. AND "abas": 34 vs 18,144, 533.6x >= 16x: gallop -> 5 left   3.88 µs
  3. AND "aam": 5 vs 95,989, 19197.8x >= 16x: gallop -> 4 left   0.71 µs
  4. AND "aal": 4 vs 99,972, 24993.0x >= 16x: gallop -> 4 left   0.58 µs
  5. BM25-score 4 match(es), sort them all   2.00 µs

Total 7.83 µs: 4 matched, returning 4.
```

## Layout

```
src/lib.rs           crate root: module list and public API
src/index.rs         Index: term dictionary, add_document, the query path, loaders
src/postings.rs      compressed posting lists (delta + varint blocks, skip table)
src/intersect.rs     merge / galloping intersection and the rule choosing between them
src/rank.rs          BM25 pieces and top-k selection
src/shard.rs         ShardedIndex: parallel build, worker pool, global stats, merge
src/tokenize.rs      allocation-free tokenizer
src/hash.rs          FxHash
src/explain.rs       query plans for `explain`
src/main.rs          interactive search REPL and `explain` command
src/bin/gencorpus.rs deterministic Zipfian corpus generator
src/bin/bench.rs     benchmark harness (throughput, latency percentiles, peak RSS)
documents/           small hand-written corpus for sanity checks
BENCHMARKS.md        every version's numbers, in order
```

## Roadmap

- [x] V1: in-memory inverted index, single-term lookup, benchmark harness
- [x] V2/V3: multi-term AND with adaptive merge/galloping intersection
- [x] V4: indexing performance (tokenizer, key allocation, hashing)
- [x] V5: BM25 relevance scoring with top-k selection
- [x] V6: `explain` query plans
- [x] V7: posting list compression (blocked delta + varint)
- [x] V8: document-partitioned shards built in parallel, global BM25 statistics,
      worker-pool fan-out with a cost-based parallel/sequential choice
- [ ] Win back V7's query cost: decode blocks only as far as needed, separate tf stream
- [ ] Sharding across processes

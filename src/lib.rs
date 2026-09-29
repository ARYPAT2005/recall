//! Recall: a full-text search engine, built from scratch.
//!
//! This is the *library*. The binaries in `src/bin/` and `src/main.rs` all
//! import it. Keeping the engine in a lib (instead of one big main.rs) is what
//! makes sharding possible later: a shard is just this struct in its own
//! process.

use std::cmp::Ordering;
use std::borrow::Cow;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

mod explain;
pub use explain::{Plan, PlanStep, PlanTerm};

/// FxHash, the hasher rustc uses internally: one rotate, xor and multiply per
/// 8 bytes. The std default (SipHash) is built to resist HashDoS from
/// attacker-chosen keys, which costs several times more per short key. Our
/// keys come from our own corpus, so that protection buys nothing here.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add(u64::from_le_bytes(*chunk));
        }
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The multiply leaves the best-mixed bits at the top, but hashbrown
        // picks buckets from the bottom bits. Rotating brings them down.
        self.hash.rotate_left(26)
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// A document's identity is just its position in `doc_names`.
///
/// u32, not usize: 4 bytes instead of 8. Posting lists are the bulk of the
/// index, so this literally halves the memory of the largest structure in the
/// program. It caps us at ~4.3 billion documents, which is fine.
pub type DocId = u32;

/// "Machine Learning, fast!" -> ["machine", "learning", "fast"]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for_each_token(text, &mut String::new(), |t| out.push(t.to_owned()));
    out
}

/// The allocation-free tokenizer behind `tokenize`: calls `f` with each
/// lowercase token. A token is a maximal run of alphanumeric chars.
///
/// Already-lowercase ASCII words (almost all of them) are passed as a slice of
/// `text` with no copy. Words with uppercase ASCII are lowercased into `buf`,
/// which is reused, so indexing a document does no per-token allocation. Only
/// non-ASCII words fall back to `str::to_lowercase`, which handles cases like
/// Greek final sigma that char-by-char lowercasing gets wrong.
pub fn for_each_token(text: &str, buf: &mut String, mut f: impl FnMut(&str)) {
    let mut emit = |word: &str, upper: bool, non_ascii: bool| {
        if non_ascii {
            f(&word.to_lowercase());
        } else if upper {
            buf.clear();
            buf.push_str(word);
            buf.make_ascii_lowercase();
            f(buf);
        } else {
            f(word);
        }
    };

    let mut start = None; // byte offset where the current token began
    let (mut upper, mut non_ascii) = (false, false);
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            start.get_or_insert(i);
            upper |= c.is_ascii_uppercase();
            non_ascii |= !c.is_ascii();
        } else if let Some(s) = start.take() {
            emit(&text[s..i], upper, non_ascii);
            (upper, non_ascii) = (false, false);
        }
    }
    if let Some(s) = start {
        emit(&text[s..], upper, non_ascii);
    }
}

/// How two posting lists get intersected. `Adaptive` is what `search` uses;
/// the other two exist so the benchmark can measure each algorithm alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Merge,
    Gallop,
    Adaptive,
}

/// Adaptive switches from merge to galloping once the longer list is at least
/// this many times the shorter one. Chosen from the ratio sweep in bench.rs,
/// and it depends on BLOCK: with 128-entry blocks the crossover moved to
/// between 16x and 32x; with 64 it's back between 8x and 16x.
pub const GALLOP_RATIO: usize = 16;

impl Strategy {
    /// The algorithm this strategy runs for lists of these lengths: Adaptive
    /// resolves to Merge or Gallop, the others to themselves.
    pub fn resolve(self, a_len: usize, b_len: usize) -> Strategy {
        let (shorter, longer) = (a_len.min(b_len), a_len.max(b_len));
        match self {
            Strategy::Adaptive if longer >= shorter.saturating_mul(GALLOP_RATIO) => Strategy::Gallop,
            Strategy::Adaptive => Strategy::Merge,
            fixed => fixed,
        }
    }
}

/// Documents present in both sorted, deduped lists, choosing the algorithm
/// from the length ratio.
pub fn intersect(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    intersect_with(a, b, Strategy::Adaptive)
}

pub fn intersect_with(a: &[DocId], b: &[DocId], strategy: Strategy) -> Vec<DocId> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    match strategy.resolve(small.len(), large.len()) {
        Strategy::Merge => intersect_merge(small, large),
        _ => intersect_gallop(small, large),
    }
}

/// Index of the first element of `s` that is >= `id`.
fn gallop_to(s: &[DocId], id: DocId) -> usize {
    gallop_by(s, |&x| x < id)
}

/// Index of the first element for which `before` is false, where `before`
/// is true for a prefix of `s`. Probes s[1], s[2], s[4]... until one is past
/// the boundary, then binary-searches that bracket, so the cost is O(log k)
/// where k is the answer: a nearby target is cheap.
fn gallop_by<T>(s: &[T], mut before: impl FnMut(&T) -> bool) -> usize {
    // Double `hi` while s[hi] is still before the boundary. When it stops,
    // the answer lies in s[hi/2 ..= hi].
    let mut hi = 1;
    while hi < s.len() && before(&s[hi]) {
        hi *= 2;
    }
    let lo = hi / 2;
    let end = (hi + 1).min(s.len());
    lo + s[lo..end].partition_point(before)
}

/// Galloping (exponential) search: for each id in `small`, gallop forward in
/// `large` from the last match. Cost is O(m log(n/m)) instead of O(m + n), so
/// "zyzzyva AND the" skips most of the 90K-entry list rather than walking it.
/// Loses to merge when the lists are similar in length: every step pays for a
/// binary search where merge would just bump a cursor.
pub fn intersect_gallop(small: &[DocId], large: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(small.len());
    // Everything in large[..base] is already known to be < the current id.
    let mut base = 0;
    for &id in small {
        base += gallop_to(&large[base..], id);
        if base == large.len() {
            break; // every remaining id in small is past the end of large
        }
        if large[base] == id {
            out.push(id);
            base += 1;
        }
    }
    out
}

/// Two cursors walk in lockstep and the smaller one advances, so this is
/// O(len(a) + len(b)) with no hashing - it only works because add_document
/// keeps every list sorted.
pub fn intersect_merge(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Entries per compressed block. See PostingList. Smaller blocks mean less to
/// decode per lookup but more skip-table entries. Measured against 128 and
/// 256: 64 makes rare-in-long-list lookups 30-45% faster for 2% more bytes.
pub const BLOCK: usize = 64;

/// Append `v` as a LEB128 varint: 7 bits per byte, high bit set on every
/// byte but the last. Values under 128 take one byte, under 16,384 two.
fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn read_varint(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut value = 0u32;
    let mut shift = 0;
    loop {
        let byte = bytes[*pos];
        *pos += 1;
        value |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            return value;
        }
        shift += 7;
    }
}

/// One term's postings: the documents containing it, ascending, and how many
/// times the term occurs in each (term frequency, which BM25 needs).
///
/// Stored compressed. Doc ids are replaced by the gap from the previous id,
/// and gaps and term frequencies are written as varints - most gaps and
/// nearly all tfs fit in one byte instead of four. Deltas alone would save
/// nothing if each gap still took a u32; the varint is what shrinks them.
///
/// Varints can't be indexed, which would kill galloping. So entries are
/// grouped into blocks of BLOCK, and a skip table holds each block's first
/// doc id and byte offset: search gallops over the skip table and decodes
/// only the blocks that can contain what it's looking for.
///
/// Search code only goes through these methods, never the raw bytes.
#[derive(Default)]
pub struct PostingList {
    len: u32,
    /// The previous doc id appended: the base for the next gap.
    last: DocId,
    /// Per block: (first doc id, byte offset of the block in `bytes`).
    skips: Vec<(DocId, u32)>,
    /// Per entry: varint(gap from previous doc in the block), varint(tf).
    /// A block's first gap is from the block's own first id, so it's 0 and
    /// the block decodes without looking at the one before it.
    bytes: Vec<u8>,
}

impl PostingList {
    /// Document frequency: how many documents contain the term.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append a document and the term's count in it. Docs arrive in
    /// ascending id order, once each, so the list stays sorted and deduped
    /// - which is exactly what intersection needs.
    fn push(&mut self, id: DocId, tf: u32) {
        debug_assert!(self.len == 0 || id > self.last, "doc ids must ascend");
        if self.len().is_multiple_of(BLOCK) {
            self.skips.push((id, self.bytes.len() as u32));
            self.last = id;
        }
        write_varint(&mut self.bytes, id - self.last);
        write_varint(&mut self.bytes, tf);
        self.last = id;
        self.len += 1;
    }

    /// Decode block `b` into the front of `docs` and `tfs`; returns its length.
    fn decode_block(&self, b: usize, docs: &mut [DocId; BLOCK], tfs: &mut [u32; BLOCK]) -> usize {
        let n = (self.len() - b * BLOCK).min(BLOCK);
        let (mut doc, offset) = self.skips[b];
        let mut pos = offset as usize;
        for i in 0..n {
            doc += read_varint(&self.bytes, &mut pos);
            docs[i] = doc;
            tfs[i] = read_varint(&self.bytes, &mut pos);
        }
        n
    }

    /// Every doc id, ascending.
    pub fn doc_ids(&self) -> Cow<'_, [DocId]> {
        let mut out = Vec::with_capacity(self.len());
        let (mut docs, mut tfs) = ([0; BLOCK], [0; BLOCK]);
        for b in 0..self.skips.len() {
            let n = self.decode_block(b, &mut docs, &mut tfs);
            out.extend_from_slice(&docs[..n]);
        }
        Cow::Owned(out)
    }

    /// The ids in `candidates` (sorted) that this list also contains.
    pub fn intersect(&self, candidates: &[DocId], strategy: Strategy) -> Vec<DocId> {
        match strategy.resolve(candidates.len(), self.len()) {
            Strategy::Merge => self.intersect_merge(candidates),
            _ => {
                let mut cursor = Cursor::new(self);
                candidates
                    .iter()
                    .copied()
                    .filter(|&id| cursor.find(id).is_some())
                    .collect()
            }
        }
    }

    /// Decode every block in order and merge against the candidates.
    fn intersect_merge(&self, candidates: &[DocId]) -> Vec<DocId> {
        let mut out = Vec::with_capacity(candidates.len().min(self.len()));
        let (mut docs, mut tfs) = ([0; BLOCK], [0; BLOCK]);
        let mut i = 0;
        for b in 0..self.skips.len() {
            if i == candidates.len() {
                break;
            }
            let n = self.decode_block(b, &mut docs, &mut tfs);
            let mut j = 0;
            while i < candidates.len() && j < n {
                match candidates[i].cmp(&docs[j]) {
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                    Ordering::Equal => {
                        out.push(docs[j]);
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
        out
    }

    /// Term frequency in each of `docs`, which must all be in this list - they
    /// are the output of intersecting with it.
    pub fn tfs_for(&self, docs: &[DocId], out: &mut Vec<u32>) {
        out.clear();
        let mut cursor = Cursor::new(self);
        for &id in docs {
            out.push(cursor.find(id).expect("tfs_for: doc not in list"));
        }
    }

    /// Heap bytes: (holding entries, allocated including unused capacity).
    pub fn heap_bytes(&self) -> (usize, usize) {
        let skip = std::mem::size_of::<(DocId, u32)>();
        (
            self.skips.len() * skip + self.bytes.len(),
            self.skips.capacity() * skip + self.bytes.capacity(),
        )
    }
}

/// Forward-only reader over a PostingList for ascending lookups. Keeps the
/// current block decoded, so consecutive lookups in one block decode it once.
struct Cursor<'a> {
    list: &'a PostingList,
    block: Option<usize>,
    pos: usize,
    n: usize,
    docs: [DocId; BLOCK],
    tfs: [u32; BLOCK],
}

impl<'a> Cursor<'a> {
    fn new(list: &'a PostingList) -> Self {
        Cursor { list, block: None, pos: 0, n: 0, docs: [0; BLOCK], tfs: [0; BLOCK] }
    }

    /// If `id` is in the list, its term frequency. Each call's id must be
    /// greater than the previous call's.
    fn find(&mut self, id: DocId) -> Option<u32> {
        // The block that could hold id is the last one starting at or before
        // it. Gallop over the skip table from the current block: the next
        // id is usually in this block or close after it.
        let from = self.block.unwrap_or(0);
        let k = gallop_by(&self.list.skips[from..], |&(first, _)| first <= id);
        if k == 0 {
            return None; // id comes before every block from here on
        }
        let b = from + k - 1;
        if self.block != Some(b) {
            self.n = self.list.decode_block(b, &mut self.docs, &mut self.tfs);
            self.block = Some(b);
            self.pos = 0;
        }
        self.pos += gallop_to(&self.docs[self.pos..self.n], id);
        if self.pos < self.n && self.docs[self.pos] == id {
            self.pos += 1;
            Some(self.tfs[self.pos - 1])
        } else {
            None
        }
    }
}

/// What an unknown query term resolves to. See `Index::matching`.
static NO_POSTINGS: PostingList = PostingList {
    len: 0,
    last: 0,
    skips: Vec::new(),
    bytes: Vec::new(),
};

/// BM25's standard parameters. K1 limits how much repeating a term can add
/// (the 10th "rust" in a doc is worth far less than the 1st). B sets how
/// strongly long documents are penalized for matching by sheer length.
pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// Inverse document frequency: rare terms count for more. This is the Lucene
/// form, ln(1 + (N - df + 0.5) / (df + 0.5)), which stays positive even for a
/// term in every document (the textbook form goes negative there).
pub fn idf(num_docs: usize, df: usize) -> f32 {
    let (n, df) = (num_docs as f32, df as f32);
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// A scored document.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub doc: DocId,
    pub score: f32,
}

/// Ordered best first: `a < b` means a ranks above b. Higher score wins, and
/// ties go to the lower doc id so results come out the same every run.
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.doc.cmp(&other.doc))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Hit {}

/// The k best hits, best first, in O(n log k) instead of sorting all n.
///
/// BinaryHeap is a max-heap, and Hit orders best first, so the heap's top is
/// the *worst* of the k kept so far. Each new hit either loses to it - one
/// comparison, the common case once the heap has filled with good hits - or
/// replaces it and sifts down in O(log k).
///
/// When k covers every hit there is nothing to select: pushing them all
/// through the heap would just be a slow heap sort, so sort directly.
pub fn top_k(hits: impl IntoIterator<Item = Hit>, k: usize) -> Vec<Hit> {
    let hits = hits.into_iter();
    if hits.size_hint().1.is_some_and(|n| n <= k) {
        let mut all: Vec<Hit> = hits.collect();
        all.sort_unstable();
        return all;
    }
    let mut heap = BinaryHeap::new();
    if k == 0 {
        return Vec::new();
    }
    for hit in hits {
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(mut worst) = heap.peek_mut() {
            if hit < *worst {
                *worst = hit;
            }
        }
    }
    heap.into_sorted_vec()
}

/// Ranked results: how many documents matched, and the best of them.
pub struct Ranked {
    pub matched: usize,
    pub hits: Vec<Hit>,
}

#[derive(Default)]
pub struct Index {
    /// Term -> its id, an index into `lists`.
    term_ids: HashMap<String, u32, FxBuildHasher>,
    /// The inverted index: term id -> its posting list.
    lists: Vec<PostingList>,
    /// Scratch for add_document: the current doc's count for each term id,
    /// and which ids it has touched. Kept here so no doc allocates them.
    pending_tf: Vec<u32>,
    touched: Vec<u32>,
    /// doc_names[id] -> the filename or title
    pub doc_names: Vec<String>,
    /// doc_lens[id] -> token count, for BM25's length normalization.
    pub doc_lens: Vec<u32>,
    /// Sum of doc_lens, so avg_doc_len is O(1) on the query path.
    total_len: u64,
}

impl Index {
    pub fn new() -> Self {
        Index::default()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_names.len()
    }

    pub fn num_terms(&self) -> usize {
        self.lists.len()
    }

    /// Total entries across all posting lists - the real size driver.
    pub fn num_postings(&self) -> usize {
        self.lists.iter().map(|l| l.len()).sum()
    }

    /// Heap bytes held by all posting lists (not the term strings or table):
    /// (holding entries, allocated including unused capacity).
    pub fn posting_bytes(&self) -> (usize, usize) {
        self.lists
            .iter()
            .map(|l| l.heap_bytes())
            .fold((0, 0), |(u, a), (lu, la)| (u + lu, a + la))
    }

    /// Average document length in tokens. BM25 needs this.
    pub fn avg_doc_len(&self) -> f64 {
        if self.doc_lens.is_empty() {
            return 0.0;
        }
        self.total_len as f64 / self.doc_lens.len() as f64
    }

    /// Every term with its posting list, in no particular order.
    pub fn terms(&self) -> impl Iterator<Item = (&str, &PostingList)> {
        self.term_ids.iter().map(|(t, &id)| (t.as_str(), &self.lists[id as usize]))
    }

    pub fn add_document(&mut self, name: String, text: &str) -> DocId {
        let id = self.doc_names.len() as DocId;
        let Index { term_ids, lists, pending_tf, touched, .. } = self;
        let mut len = 0u32;

        // Count each term in this doc first, then append one (doc, tf) entry
        // per distinct term. Compressed lists are append-only, so a repeat
        // can't go back and bump a tf that's already been encoded.
        for_each_token(text, &mut String::new(), |word| {
            len += 1;
            // get first: entry() needs an owned String key, which would
            // allocate once per token even though nearly every term already
            // exists. Only a brand-new term pays for the allocation.
            let term = match term_ids.get(word) {
                Some(&term) => term,
                None => {
                    let term = lists.len() as u32;
                    term_ids.insert(word.to_owned(), term);
                    lists.push(PostingList::default());
                    pending_tf.push(0);
                    term
                }
            };
            let count = &mut pending_tf[term as usize];
            if *count == 0 {
                touched.push(term);
            }
            *count += 1;
        });
        for term in touched.drain(..) {
            let count = &mut pending_tf[term as usize];
            lists[term as usize].push(id, *count);
            *count = 0;
        }

        self.doc_names.push(name);
        self.doc_lens.push(len);
        self.total_len += len as u64;
        id
    }

    /// The posting list for one term, if any document contains it.
    pub fn postings_for(&self, term: &str) -> Option<&PostingList> {
        self.term_ids.get(term).map(|&id| &self.lists[id as usize])
    }

    /// Boolean AND: documents containing every term in the query, ascending.
    /// The query goes through the same tokenizer as the documents, so "Machine
    /// Learning!" and "machine learning" are the same search.
    pub fn search(&self, query: &str) -> Vec<DocId> {
        self.search_with(query, Strategy::Adaptive)
    }

    pub fn search_with(&self, query: &str, strategy: Strategy) -> Vec<DocId> {
        self.matching(query, strategy, None).1
    }

    /// The k most relevant documents containing every query term, by BM25.
    pub fn search_ranked(&self, query: &str, k: usize) -> Ranked {
        let (lists, docs) = self.matching(query, Strategy::Adaptive, None);
        self.rank(&lists, &docs, k)
    }

    /// Run a ranked query and report every decision the engine made: term
    /// order, each intersection's algorithm and why, and where time went.
    pub fn explain(&self, query: &str, k: usize) -> Plan {
        // Run it once and throw that plan away. The first run pays one-time
        // costs - cold posting lists, and the process's first clock read,
        // which on macOS resolves the timer symbol lazily (~13 µs) - that
        // would otherwise land on whichever step happens to go first.
        self.explain_once(query, k);
        self.explain_once(query, k)
    }

    fn explain_once(&self, query: &str, k: usize) -> Plan {
        let mut plan = Plan {
            query: query.to_owned(),
            num_docs: self.num_docs(),
            k,
            ..Plan::default()
        };
        let (lists, docs) = self.matching(query, Strategy::Adaptive, Some(&mut plan));
        let start = Instant::now();
        let ranked = self.rank(&lists, &docs, k);
        plan.rank_time = start.elapsed();
        plan.matched = ranked.matched;
        plan.hits = ranked.hits;
        plan
    }

    fn rank(&self, lists: &[&PostingList], docs: &[DocId], k: usize) -> Ranked {
        let scores = self.score(lists, docs);
        let hits = docs.iter().zip(&scores).map(|(&doc, &score)| Hit { doc, score });
        Ranked {
            matched: docs.len(),
            hits: top_k(hits, k),
        }
    }

    /// The query's posting lists (rarest first) and the docs in all of them.
    /// With `plan`, also records each decision and its timing - the only time
    /// this path reads the clock.
    fn matching(
        &self,
        query: &str,
        strategy: Strategy,
        mut plan: Option<&mut Plan>,
    ) -> (Vec<&PostingList>, Vec<DocId>) {
        let started = plan.is_some().then(Instant::now);
        let mut terms = tokenize(query);
        // "rust rust" is one term. Scoring it twice would double its weight.
        terms.sort_unstable();
        terms.dedup();

        // An unknown term gets an empty list instead of a special case: it
        // sorts first as the rarest term, and the intersection comes out
        // empty, which is exactly what AND means.
        let mut lists: Vec<(String, &PostingList)> = terms
            .into_iter()
            .map(|t| {
                let list = self.postings_for(&t).unwrap_or(&NO_POSTINGS);
                (t, list)
            })
            .collect();
        // Shortest first: the result can never outgrow the shortest list, so
        // starting there bounds every later intersection by the rarest term
        // instead of dragging the longest list through each step. The sort is
        // stable, so equally common terms stay in alphabetical order.
        lists.sort_by_key(|(_, l)| l.len());

        if let (Some(plan), Some(started)) = (plan.as_deref_mut(), started) {
            plan.terms = lists
                .iter()
                .map(|(t, l)| PlanTerm {
                    term: t.clone(),
                    df: l.len(),
                    idf: idf(self.num_docs(), l.len()),
                })
                .collect();
            plan.lookup_time = started.elapsed();
        }
        let Some(&(_, first)) = lists.first() else {
            return (Vec::new(), Vec::new());
        };

        let mut docs = first.doc_ids();
        for (term, list) in &lists[1..] {
            if docs.is_empty() {
                break;
            }
            let start = plan.is_some().then(Instant::now);
            let next = list.intersect(&docs, strategy);
            if let (Some(plan), Some(start)) = (plan.as_deref_mut(), start) {
                plan.steps.push(PlanStep {
                    term: term.clone(),
                    candidates: docs.len(),
                    df: list.len(),
                    algorithm: strategy.resolve(docs.len(), list.len()),
                    matched: next.len(),
                    time: start.elapsed(),
                });
            }
            docs = Cow::Owned(next);
        }
        let docs = docs.into_owned();
        (lists.into_iter().map(|(_, l)| l).collect(), docs)
    }

    /// BM25 score for each of `docs`, which contain every term in `lists`:
    ///
    ///   sum over terms of  idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
    ///
    /// tf saturates (k1 caps what repetition adds), and a doc longer than
    /// average needs more occurrences for the same score (b).
    fn score(&self, lists: &[&PostingList], docs: &[DocId]) -> Vec<f32> {
        let avg_len = self.avg_doc_len() as f32;
        // The length part of the denominator depends only on the doc, so
        // compute it once per doc rather than once per (doc, term).
        let norms: Vec<f32> = docs
            .iter()
            .map(|&d| {
                let len = self.doc_lens[d as usize] as f32;
                BM25_K1 * (1.0 - BM25_B + BM25_B * len / avg_len)
            })
            .collect();

        let mut scores = vec![0.0f32; docs.len()];
        let mut tfs = Vec::with_capacity(docs.len());
        for list in lists {
            let idf = idf(self.num_docs(), list.len());
            list.tfs_for(docs, &mut tfs);
            for ((score, &tf), &norm) in scores.iter_mut().zip(&tfs).zip(&norms) {
                let tf = tf as f32;
                *score += idf * tf * (BM25_K1 + 1.0) / (tf + norm);
            }
        }
        scores
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Index {
        let mut idx = Index::new();
        idx.add_document("a".into(), "machine learning with rust");
        idx.add_document("b".into(), "machine shop");
        idx.add_document("c".into(), "deep learning, machine learning!");
        idx.add_document("d".into(), "rust learning");
        idx
    }

    const STRATEGIES: [Strategy; 3] = [Strategy::Merge, Strategy::Gallop, Strategy::Adaptive];

    #[test]
    fn intersect_keeps_common_ids() {
        for s in STRATEGIES {
            assert_eq!(intersect_with(&[1, 3, 5, 8, 12], &[3, 4, 8, 20], s), vec![3, 8]);
            assert_eq!(intersect_with(&[1, 2], &[3, 4], s), Vec::<DocId>::new());
            assert_eq!(intersect_with(&[], &[1, 2], s), Vec::<DocId>::new());
            assert_eq!(intersect_with(&[1, 2], &[], s), Vec::<DocId>::new());
        }
    }

    #[test]
    fn gallop_handles_edges_of_the_long_list() {
        let large: Vec<DocId> = (0..1000).map(|i| i * 3).collect();
        // First element, last element, past the end, before the start.
        assert_eq!(intersect_gallop(&[0], &large), vec![0]);
        assert_eq!(intersect_gallop(&[2997], &large), vec![2997]);
        assert_eq!(intersect_gallop(&[5000], &large), Vec::<DocId>::new());
        assert_eq!(intersect_gallop(&[1, 2, 3], &large), vec![3]);
        // Consecutive matches must not skip each other.
        assert_eq!(intersect_gallop(&[3, 6, 9], &large), vec![3, 6, 9]);
    }

    #[test]
    fn gallop_matches_merge_on_random_lists() {
        // xorshift, fixed seed: same cases every run.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            // Vary density independently so ratios range from 1:1 to ~1:1000.
            let universe = 1 + next() % 5000;
            let (da, db) = (1 + next() % 1000, 1 + next() % 1000);
            let mut list = |density: u64| -> Vec<DocId> {
                (0..universe as DocId).filter(|_| next() % 1000 < density).collect()
            };
            let a = list(da);
            let b = list(db);
            let expected = intersect_merge(&a, &b);
            assert_eq!(intersect_with(&a, &b, Strategy::Gallop), expected);
            assert_eq!(intersect_with(&b, &a, Strategy::Gallop), expected);
            assert_eq!(intersect(&a, &b), expected);
        }
    }

    #[test]
    fn search_is_boolean_and() {
        let idx = sample();
        assert_eq!(idx.search("machine learning"), vec![0, 2]);
        assert_eq!(idx.search("rust learning machine"), vec![0]);
        assert_eq!(idx.search("machine"), vec![0, 1, 2]);
    }

    #[test]
    fn search_tokenizes_the_query() {
        let idx = sample();
        assert_eq!(idx.search("  Machine, LEARNING!\n"), vec![0, 2]);
        assert_eq!(idx.search("learning learning"), vec![0, 2, 3]);
    }

    fn index(docs: &[&str]) -> Index {
        let mut idx = Index::new();
        for (i, text) in docs.iter().enumerate() {
            idx.add_document(format!("d{i}"), text);
        }
        idx
    }

    fn ranked_ids(idx: &Index, query: &str) -> Vec<DocId> {
        idx.search_ranked(query, 10).hits.iter().map(|h| h.doc).collect()
    }

    #[test]
    fn term_frequencies_are_counted_per_doc() {
        let idx = sample();
        let list = idx.postings_for("learning").unwrap();
        assert_eq!(list.len(), 3); // df counts documents, not occurrences
        let mut tfs = Vec::new();
        list.tfs_for(&[0, 2, 3], &mut tfs);
        assert_eq!(tfs, vec![1, 2, 1]); // doc c says "learning" twice
    }

    #[test]
    fn bm25_matches_hand_computed_score() {
        // N = 2, df(rust) = 1, avg_len = 1.5, doc 0 has len 1 and tf 1.
        // idf  = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln 2
        // norm = 1.2 * (1 - 0.75 + 0.75 * 1 / 1.5) = 0.9
        // score = ln 2 * 1 * 2.2 / (1 + 0.9)
        let idx = index(&["rust", "go go"]);
        let hits = idx.search_ranked("rust", 10).hits;
        let expected = 2f32.ln() * 2.2 / 1.9;
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - expected).abs() < 1e-6, "{} vs {expected}", hits[0].score);
    }

    #[test]
    fn bm25_rewards_term_frequency_and_short_docs() {
        // More occurrences in docs of equal length ranks higher...
        let idx = index(&["rust go go", "rust rust go"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
        // ...and the same count in a shorter doc ranks higher.
        let idx = index(&["rust a b c d e", "rust a"]);
        assert_eq!(ranked_ids(&idx, "rust"), vec![1, 0]);
    }

    #[test]
    fn bm25_weights_rare_terms_more() {
        // "machine" is in 3 docs, "rust" in 2. Docs 0 and 1 both match the
        // query with the same length; doc 1 has more of the rarer term.
        let idx = index(&["machine machine rust", "machine rust rust", "machine"]);
        assert_eq!(ranked_ids(&idx, "machine rust"), vec![1, 0]);
    }

    #[test]
    fn ranking_is_top_k_with_deterministic_ties() {
        let idx = index(&["rust", "rust", "rust rust", "rust"]);
        let r = idx.search_ranked("rust", 2);
        assert_eq!(r.matched, 4);
        // Doc 2 scores highest; docs 0, 1 and 3 tie, and the lowest id wins.
        assert_eq!(r.hits.iter().map(|h| h.doc).collect::<Vec<_>>(), vec![2, 0]);
        assert!(idx.search_ranked("rust", 0).hits.is_empty());
    }

    #[test]
    fn top_k_equals_full_sort() {
        let hits: Vec<Hit> = (0..500)
            .map(|i| Hit { doc: i, score: ((i * 7919) % 97) as f32 })
            .collect();
        let mut sorted = hits.clone();
        sorted.sort();
        for k in [1, 5, 10, 97, 500, 1000] {
            let got: Vec<DocId> = top_k(hits.iter().copied(), k).iter().map(|h| h.doc).collect();
            let want: Vec<DocId> = sorted.iter().take(k).map(|h| h.doc).collect();
            assert_eq!(got, want, "k = {k}");
        }
    }

    #[test]
    fn repeated_query_terms_count_once() {
        let idx = sample();
        let once = idx.search_ranked("rust", 10).hits;
        let twice = idx.search_ranked("rust RUST rust", 10).hits;
        assert_eq!(once, twice);
        assert_eq!(once[0].score, twice[0].score);
    }

    #[test]
    fn varint_round_trips_edge_values() {
        let values = [0, 1, 127, 128, 16_383, 16_384, 2_097_151, 2_097_152, u32::MAX];
        let mut bytes = Vec::new();
        for &v in &values {
            write_varint(&mut bytes, v);
        }
        // 1 + 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 bytes
        assert_eq!(bytes.len(), 22);
        let mut pos = 0;
        for &v in &values {
            assert_eq!(read_varint(&bytes, &mut pos), v);
        }
        assert_eq!(pos, bytes.len());
    }

    /// A list of `n` entries with gaps from tiny to huge (5-byte varints) and
    /// assorted tfs, plus the raw (docs, tfs) it should decode to.
    fn compressed(n: usize, seed: u64) -> (PostingList, Vec<DocId>, Vec<u32>) {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut docs, mut tfs) = (Vec::new(), Vec::new());
        let mut doc: DocId = (next() % 50) as DocId;
        for _ in 0..n {
            docs.push(doc);
            tfs.push(1 + (next() % 3 == 0) as u32 * (next() % 300) as u32);
            let gap = match next() % 20 {
                0 => 1 + next() % 5_000_000, // multi-byte varint
                1..=9 => 1,
                _ => 1 + next() % 100,
            };
            doc = doc.saturating_add(gap as DocId);
            if doc == DocId::MAX {
                break;
            }
        }
        let mut list = PostingList::default();
        for (&d, &t) in docs.iter().zip(&tfs) {
            list.push(d, t);
        }
        (list, docs, tfs)
    }

    #[test]
    fn compressed_list_round_trips_across_blocks() {
        for (n, seed) in [(1, 1), (BLOCK - 1, 2), (BLOCK, 3), (BLOCK + 1, 4), (5_000, 5)] {
            let (list, docs, tfs) = compressed(n, seed);
            assert_eq!(list.len(), docs.len());
            assert_eq!(&*list.doc_ids(), &docs[..]);
            let mut got = Vec::new();
            list.tfs_for(&docs, &mut got);
            assert_eq!(got, tfs);
            // A sparse subset: every 37th entry, so lookups skip blocks.
            let subset: Vec<DocId> = docs.iter().step_by(37).copied().collect();
            let want: Vec<u32> = tfs.iter().step_by(37).copied().collect();
            list.tfs_for(&subset, &mut got);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn compressed_intersect_matches_uncompressed() {
        let (list, docs, _) = compressed(3_000, 99);
        let max = *docs.last().unwrap();
        let mut state = 7u64;
        for density in [1u64, 10, 100, 1_000, 10_000] {
            // Candidates: some in the list, some between its ids, some past
            // either end.
            let mut cands: Vec<DocId> = docs
                .iter()
                .filter(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 33) % 10_000 < density
                })
                .flat_map(|&d| [d.saturating_sub(1), d, d + 1])
                .collect();
            cands.push(max + 10);
            cands.sort_unstable();
            cands.dedup();
            let want = intersect_merge(&cands, &docs);
            for s in STRATEGIES {
                assert_eq!(list.intersect(&cands, s), want, "density {density}, {s:?}");
            }
        }
    }

    #[test]
    fn explain_reports_order_steps_and_same_results_as_search() {
        let idx = sample();
        let plan = idx.explain("machine learning rust", 10);
        // rust (df 2) first; learning and machine (df 3) tie, alphabetical.
        let order: Vec<&str> = plan.terms.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(order, vec!["rust", "learning", "machine"]);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!((plan.steps[0].candidates, plan.steps[0].df), (2, 3));
        assert_eq!(plan.steps[0].algorithm, Strategy::Merge); // 1.5x < 16x
        assert_eq!(plan.hits, idx.search_ranked("machine learning rust", 10).hits);
        assert_eq!(plan.matched, 1);
    }

    #[test]
    fn explain_picks_gallop_for_skewed_lists() {
        let mut docs = vec!["common rare"];
        docs.extend(std::iter::repeat_n("common", 40));
        let plan = index(&docs).explain("common rare", 10);
        assert_eq!(plan.steps[0].algorithm, Strategy::Gallop); // 41 vs 1
        assert!(plan.to_string().contains("gallop"));
    }

    #[test]
    fn explain_stops_at_an_unknown_term() {
        let plan = sample().explain("machine zebra learning", 10);
        assert_eq!(plan.terms[0].term, "zebra");
        assert_eq!(plan.terms[0].df, 0);
        assert!(plan.steps.is_empty()); // nothing left to intersect
        assert_eq!(plan.matched, 0);
        assert!(plan.to_string().contains("never read"));
    }

    #[test]
    fn unknown_term_or_empty_query_matches_nothing() {
        let idx = sample();
        assert!(idx.search("machine zebra").is_empty());
        assert!(idx.search("").is_empty());
        assert!(idx.search("!!!").is_empty());
    }
}

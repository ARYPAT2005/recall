//! Compressed posting lists: blocked delta + varint encoding with a skip table.

use crate::intersect::{gallop_by, gallop_to};
use crate::{DocId, Strategy};
use std::cmp::Ordering;

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
    pub(crate) fn push(&mut self, id: DocId, tf: u32) {
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
    pub fn doc_ids(&self) -> Vec<DocId> {
        let mut out = Vec::with_capacity(self.len());
        let (mut docs, mut tfs) = ([0; BLOCK], [0; BLOCK]);
        for b in 0..self.skips.len() {
            let n = self.decode_block(b, &mut docs, &mut tfs);
            out.extend_from_slice(&docs[..n]);
        }
        out
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
pub(crate) static NO_POSTINGS: PostingList = PostingList {
    len: 0,
    last: 0,
    skips: Vec::new(),
    bytes: Vec::new(),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intersect::intersect_merge;

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
            assert_eq!(list.doc_ids(), docs);
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
            for s in Strategy::ALL {
                assert_eq!(list.intersect(&cands, s), want, "density {density}, {s:?}");
            }
        }
    }
}

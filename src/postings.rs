//! Compressed posting lists.

use crate::intersect::{gallop_by, gallop_to};
use crate::{DocId, Strategy};
use std::cmp::Ordering;

/// Entries per compressed block.
pub const BLOCK: usize = 64;

/// LEB128: 7 bits per byte, high bit set if more bytes follow.
fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Reads one varint at `pos` and moves `pos` past it.
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

/// One term's doc ids and counts: gaps and counts as varints, in blocks with a skip table.
#[derive(Default)]
pub struct PostingList {
    len: u32,
    // previous doc id added, the base for the next gap
    last: DocId,
    // per block: (first doc id, byte offset)
    skips: Vec<(DocId, u32)>,
    // per entry: varint gap, varint count
    bytes: Vec<u8>,
}

impl PostingList {
    /// Number of documents containing the term.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adds a document. Ids must be increasing.
    pub(crate) fn push(&mut self, id: DocId, tf: u32) {
        debug_assert!(self.len == 0 || id > self.last, "doc ids must ascend");
        // Start a new block.
        if self.len().is_multiple_of(BLOCK) {
            self.skips.push((id, self.bytes.len() as u32));
            self.last = id;
        }
        write_varint(&mut self.bytes, id - self.last);
        write_varint(&mut self.bytes, tf);
        self.last = id;
        self.len += 1;
    }

    /// Decodes block `b` into `docs` and `tfs` and returns how many entries it had.
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

    /// All doc ids, decoded.
    pub fn doc_ids(&self) -> Vec<DocId> {
        let mut out = Vec::with_capacity(self.len());
        let (mut docs, mut tfs) = ([0; BLOCK], [0; BLOCK]);
        for b in 0..self.skips.len() {
            let n = self.decode_block(b, &mut docs, &mut tfs);
            out.extend_from_slice(&docs[..n]);
        }
        out
    }

    /// The ids in `candidates` that are also in this list.
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

    /// Decodes every block and merges.
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

    /// The term's count in each of `docs`. Every doc must be in the list.
    pub fn tfs_for(&self, docs: &[DocId], out: &mut Vec<u32>) {
        out.clear();
        let mut cursor = Cursor::new(self);
        for &id in docs {
            out.push(cursor.find(id).expect("tfs_for: doc not in list"));
        }
    }

    /// (bytes used, bytes allocated).
    pub fn heap_bytes(&self) -> (usize, usize) {
        let skip = std::mem::size_of::<(DocId, u32)>();
        (
            self.skips.len() * skip + self.bytes.len(),
            self.skips.capacity() * skip + self.bytes.capacity(),
        )
    }
}

/// Walks a list forward, keeping the current block decoded.
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

    /// The count for `id` if it's in the list. Ids must increase between calls.
    fn find(&mut self, id: DocId) -> Option<u32> {
        let from = self.block.unwrap_or(0);
        let k = gallop_by(&self.list.skips[from..], |&(first, _)| first <= id);
        if k == 0 {
            return None;
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

/// Empty list used for words that aren't in the index.
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
        assert_eq!(bytes.len(), 22);
        let mut pos = 0;
        for &v in &values {
            assert_eq!(read_varint(&bytes, &mut pos), v);
        }
        assert_eq!(pos, bytes.len());
    }

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
                0 => 1 + next() % 5_000_000,
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

//! FxHash, a fast non-cryptographic hasher for the term dictionary.

use std::hash::{BuildHasherDefault, Hasher};

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

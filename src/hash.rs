use bytemuck::{Pod, Zeroable};

#[cfg(not(feature = "solana"))]
use sha3::Digest;

pub const HASH_BYTES: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Default, Pod, Zeroable)]
pub struct Hash {
   pub(crate) value: [u8; 32]
}

impl From<Hash> for [u8; HASH_BYTES] {
    fn from(from: Hash) -> Self {
        from.value
    }
}

impl From<[u8; HASH_BYTES]> for Hash {
    fn from(from: [u8; 32]) -> Self {
        Self { value: from }
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.value
    }
}

impl Hash {
    pub const LEN: usize = HASH_BYTES;

    pub fn new(hash_slice: &[u8]) -> Self {
        Hash { value: <[u8; HASH_BYTES]>::try_from(hash_slice).unwrap() }
    }

    pub const fn new_from_array(hash_array: [u8; HASH_BYTES]) -> Self {
        Self { value: hash_array }
    }

    pub fn to_bytes(self) -> [u8; HASH_BYTES] {
        self.value
    }
}

#[cfg(feature = "solana")]
#[inline(always)]
pub fn hashv(data: &[&[u8]]) -> Hash {
    let res = solana_program::keccak::hashv(data);
    Hash::new_from_array(res.to_bytes())
}

#[cfg(not(feature = "solana"))]
#[inline(always)]
pub fn hashv(data: &[&[u8]]) -> Hash {
    let mut hasher = sha3::Keccak256::new();
    for d in data {
        hasher.update(d);
    }
    Hash::new_from_array(hasher.finalize().into())
}

#[cfg(feature = "solana")]
#[inline(always)]
pub fn hash(data: &[u8]) -> Hash {
    let res = solana_program::keccak::hash(data);
    Hash::new_from_array(res.to_bytes())
}

#[cfg(not(feature = "solana"))]
#[inline(always)]
pub fn hash(data: &[u8]) -> Hash {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(data);
    Hash::new_from_array(hasher.finalize().into())
}

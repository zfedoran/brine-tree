use sha2::{Digest, Sha256};
use super::hash::Hash;
use super::error::{ProgramError, ProgramResult};

#[inline]
pub fn hash(data: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    Hash::new_from_array(hasher.finalize().into())
}

#[inline]
pub fn hashv(data: &[&[u8]]) -> Hash {
    let mut hasher = Sha256::new();
    for d in data {
        hasher.update(d);
    }
    Hash::new_from_array(hasher.finalize().into())
}

#[inline]
pub fn check_condition(condition: bool, msg: &'static str) -> ProgramResult {
    if condition {
        Ok(())
    } else {
        Err(ProgramError::Custom(msg))
    }
}

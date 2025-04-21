use super::hash::Hash;
use super::error::{ProgramError, ProgramResult};

#[inline]
pub fn hash(data: &[u8]) -> Hash {
    Hash::new_from_array(hashv(&[data]).into())
}

#[inline]
pub fn hashv(data: &[&[u8]]) -> Hash {
    let res = solana_program::hash::hashv(data);
    Hash::new_from_array(res.to_bytes())
}

#[inline]
pub fn check_condition(condition: bool, msg: &'static str) -> ProgramResult {
    if condition {
        Ok(())
    } else {
        Err(ProgramError::Custom(msg))
    }
}

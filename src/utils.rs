use super::error::{BrineTreeError, ProgramResult};

#[inline]
pub fn check_condition(condition: bool, err: BrineTreeError) -> ProgramResult {
    if condition {
        Ok(())
    } else {
        Err(err)
    }
}

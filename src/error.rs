#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramError {
    InvalidArgument,
    Custom(&'static str),
}

pub type ProgramResult = Result<(), ProgramError>;

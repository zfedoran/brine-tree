#![cfg_attr(not(feature = "std"), no_std)]

pub mod error;
pub mod hash;
pub mod utils;
pub mod tree;
pub mod subtree;

pub use tree::{MerkleTree, verify, get_merkle_proof};
pub use subtree::{Subtree, get_cached_merkle_proof};
pub use hash::{Hash, Leaf};
pub use error::BrineTreeError;

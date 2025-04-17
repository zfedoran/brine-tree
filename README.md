# brine-tree

![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)
[![crates.io](https://img.shields.io/crates/v/brine-tree.svg?style=flat)](https://crates.io/crates/brine-tree)

A fast, low-overhead, Merkle tree library for the Solana programs.

![image](https://github.com/user-attachments/assets/6c34a34b-644f-4248-bf78-d7c43d1e18f6)


---

## ✨ Features

- Supports insertion, removal, and replacement of leaves
- Validates inclusion proofs efficiently
- Zero-copy compatibility with `bytemuck` for fast serialization

---

## 🧱 Use Cases

- State compression for large datasets
- Whitelist or access control verification
- Off-chain data integrity checks
- Cross-chain state proofs
- Decentralized identity claims
- Oracle data validation

---

## 🚀 Quick Start

```rust
use brine_tree::{ MerkleTree, Hash };

let seeds: &[&[u8]] = &[b"seed"];
let mut tree: MerkleTree<3> = MerkleTree::new(seeds);
let value = Hash::new_from_array([1; 32]);

tree.try_insert(value)?;
let proof = tree.get_merkle_proof(&[value], 0);
assert!(tree.contains(&proof, value));
```

Returns `Ok(())` for successful operations or `Err(ProgramError)` if invalid.

### But why?

**Q: Why not use an off-chain Merkle tree?**  
**A:** Solana programs often need to verify inclusion or manage state on-chain efficiently. Off-chain Merkle trees require additional infrastructure and trust assumptions. 

**Q: Why not use something else?**  
**A:** There definitely are a few [other](https://github.com/anza-xyz/agave/blob/master/merkle-tree/src/merkle_tree.rs) implementations worth looking into, like [concurrent merkle tree](https://github.com/solana-labs/solana-program-library/blob/master/libraries/concurrent-merkle-tree/src/concurrent_merkle_tree.rs), but this one is simple and easy to work with. This crate, brine-tree, provides:

- Simple, on-chain Merkle tree construction and verification
- Support for dynamic updates (insert, remove, replace)
- Low compute unit (CU) consumption
- Can be stored in account state

## Contributing

Contributions are welcome! Please open issues or PRs on the GitHub repo.


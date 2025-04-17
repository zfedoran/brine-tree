# brine-tree

![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)
[![crates.io](https://img.shields.io/crates/v/brine-tree.svg?style=flat)](https://crates.io/crates/brine-tree)

A fast, low-overhead, Merkle tree library for the Solana SVM programs.

---

## ✨ Features

- Constructs and verifies Merkle trees **within the program**, at run-time
- Supports insertion, removal, and replacement of leaves
- Validates inclusion proofs efficiently
- Zero-copy compatibility with `bytemuck` for fast serialization

---

## 🧱 Use Cases

- Whitelist or access control verification
- State compression for large datasets
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
**A:** Solana programs often need to verify inclusion or manage state on-chain efficiently. Off-chain Merkle trees require additional infrastructure and trust assumptions. This crate, brine-tree, provides:

- On-chain Merkle tree construction and verification
- Support for dynamic updates (insert, remove, replace)
- Low compute unit (CU) consumption
- Seamless integration with Solana programs

## Security

This library has been audited by `Ottersec` twice as part of two seperate programs. This version was pulled from the `code-vm`, which was written and maintained by the same author as this crate.

## Contributing

Contributions are welcome! Please open issues or PRs on the GitHub repo.


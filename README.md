# brine-tree

![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)
[![crates.io](https://img.shields.io/crates/v/brine-tree.svg?style=flat)](https://crates.io/crates/brine-tree)

A fast, low-overhead, Merkle tree library for the Solana programs.

![image](https://github.com/user-attachments/assets/6c34a34b-644f-4248-bf78-d7c43d1e18f6)


---

## ✨ Features

- Support for add, remove, replace
- Low compute unit (CU) consumption
- Can be stored in account state
- Zero-copy

---

## 🚀 Quick Start

```rust
use brine_tree::{MerkleTree, Leaf};

fn main() {
    const TREE_DEPTH: usize = 18;

    let mut tree = MerkleTree::<{TREE_DEPTH}>::new(&[b"empty_leaf_seed"]);
    let data = &[b"hello", b"world"];
    let leaf = Leaf::new(data);

    tree.try_add_leaf(leaf)?;

    // Off-chain proof generation

    let db = &[leaf, ...]; // your database of leaves
    let leaf_index = 0; // index of the leaf you want to prove

    let proof = tree.get_merkle_proof(db, leaf_index)?;
    
    assert!(tree.contains(&proof, data));

    Ok(())
}
```

Returns `Ok(())` for successful operations or `Err(ProgramError)` if invalid.

---

## 🧱 Use Cases

- State compression for large datasets
- Whitelist or access control verification
- Off-chain data integrity checks
- Cross-chain state proofs
- Decentralized identity claims
- Oracle data validation

---

### But why?

**Q: Why not use an off-chain Merkle tree?**  
**A:** Solana programs often need to verify inclusion or manage state on-chain efficiently. Off-chain Merkle trees require additional infrastructure and trust assumptions. 

**Q: Why not use something else?**  
**A:** There definitely are a few [other](https://github.com/anza-xyz/agave/blob/master/merkle-tree/src/merkle_tree.rs) implementations worth looking into, like [concurrent merkle tree](https://github.com/solana-labs/solana-program-library/blob/master/libraries/concurrent-merkle-tree/src/concurrent_merkle_tree.rs), but this one is simple and easy to work with. 

---

## 🙌 Contributing

Contributions are welcome! Please open issues or PRs on the GitHub repo.


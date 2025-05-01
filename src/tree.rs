use bytemuck::{ Pod, Zeroable };
use super::hash::{ Hash, Leaf, hashv };
use super::error::{ BrineTreeError, ProgramResult };
use super::utils::check_condition;

#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MerkleTree<const N: usize> {
    root: Hash,
    filled_subtrees: [Hash; N],
    zero_values: [Hash; N],
    next_index: u64,
}

unsafe impl<const N: usize> Zeroable for MerkleTree<N> {}
unsafe impl<const N: usize> Pod for MerkleTree<N> {}

impl<const N: usize> MerkleTree<N> {
    pub const fn get_depth(&self) -> u8 {
        N as u8
    }

    pub const fn get_size() -> usize {
        core::mem::size_of::<Self>()
    }

    pub fn get_root(&self) -> Hash {
        self.root
    }

    pub fn get_empty_leaf(&self) -> Leaf {
        self.zero_values[0].as_leaf()
    }

    pub fn new(seeds: &[&[u8]]) -> Self {
        let zeros = Self::calc_zeros(seeds);
        Self {
            next_index: 0,
            root: zeros[N - 1],
            filled_subtrees: zeros,
            zero_values: zeros,
        }
    }

    pub fn init(&mut self, seeds: &[&[u8]]) {
        let zeros = Self::calc_zeros(seeds);
        self.next_index = 0;
        self.root = zeros[N - 1];
        self.filled_subtrees = zeros;
        self.zero_values = zeros;
    }

    fn calc_zeros(seeds: &[&[u8]]) -> [Hash; N] {
        let mut zeros: [Hash; N] = [Hash::default(); N];
        let mut current = hashv(seeds);

        for i in 0..N {
            zeros[i] = current;
            current = hashv(&[b"NODE".as_ref(), current.as_ref(), current.as_ref()]);
        }

        zeros
    }

    pub fn try_add(&mut self, data: &[&[u8]]) -> ProgramResult {
        let leaf = Leaf::new(data);
        self.try_add_leaf(leaf)
    }

    pub fn try_add_leaf(&mut self, leaf: Leaf) -> ProgramResult {
        check_condition(
            self.next_index < (1u64 << N),
            BrineTreeError::TreeFull,
        )?;

        let mut current_index = self.next_index;
        let mut current_hash = Hash::from(leaf);
        let mut left;
        let mut right;

        for i in 0..N {
            if current_index % 2 == 0 {
                left = current_hash;
                right = self.zero_values[i];
                self.filled_subtrees[i] = current_hash;
            } else {
                left = self.filled_subtrees[i];
                right = current_hash;
            }

            current_hash = hash_left_right(left, right);
            current_index /= 2;
        }

        self.root = current_hash;
        self.next_index += 1;

        Ok(())
    }

    pub fn try_remove(&mut self, proof: &[Hash], data: &[&[u8]]) -> ProgramResult {
        let leaf = Leaf::new(data);
        self.try_remove_leaf(proof, leaf)
    }

    pub fn try_remove_leaf(&mut self, proof: &[Hash], leaf: Leaf) -> ProgramResult {
        self.check_length(proof)?;
        self.try_replace_leaf(proof, leaf, self.get_empty_leaf())
    }

    pub fn try_replace(&mut self, proof: &[Hash], original_data: &[&[u8]], new_data: &[&[u8]]) -> ProgramResult {
        let original_leaf = Leaf::new(original_data);
        let new_leaf = Leaf::new(new_data);
        self.try_replace_leaf(proof, original_leaf, new_leaf)
    }

    pub fn try_replace_leaf(&mut self, proof: &[Hash], original_leaf: Leaf, new_leaf: Leaf) -> ProgramResult {
        self.check_length(proof)?;

        let original_path = compute_path(proof, original_leaf);
        let new_path = compute_path(proof, new_leaf);

        check_condition(
            is_valid_path(&original_path, self.root),
            BrineTreeError::InvalidProof,
        )?;

        for i in 0..N {
            if original_path[i] == self.filled_subtrees[i] {
                self.filled_subtrees[i] = new_path[i];
            }
        }

        self.root = *new_path.last().unwrap();

        Ok(())
    }

    pub fn contains(&self, proof: &[Hash], data: &[&[u8]]) -> bool {
        let leaf = Leaf::new(data);
        self.contains_leaf(proof, leaf)
    }

    pub fn contains_leaf(&self, proof: &[Hash], leaf: Leaf) -> bool {
        if let Err(_) = self.check_length(proof) {
            return false;
        }

        let root = self.get_root();
        is_valid_leaf(proof, root, leaf)
    }

    fn check_length(&self, proof: &[Hash]) -> Result<(), BrineTreeError> {
        check_condition(proof.len() == N, BrineTreeError::ProofLength)
    }

    #[cfg(not(feature = "solana"))]
    fn hash_pairs(pairs: Vec<Hash>) -> Vec<Hash> {
        let mut res = Vec::with_capacity(pairs.len() / 2);

        for i in (0..pairs.len()).step_by(2) {
            let left = pairs[i];
            let right = pairs[i + 1];

            let hashed = hash_left_right(left, right);
            res.push(hashed);
        }

        res
    }

    #[cfg(not(feature = "solana"))]
    pub fn get_merkle_proof(&self, values: &[Leaf], index: usize) -> Vec<Hash> {
        let mut layers = Vec::with_capacity(N);
        let mut current_layer: Vec<Hash> = values.iter().map(|leaf| Hash::from(*leaf)).collect();

        for i in 0..N {
            if current_layer.len() % 2 != 0 {
                current_layer.push(self.zero_values[i]);
            }

            layers.push(current_layer.clone());
            current_layer = Self::hash_pairs(current_layer);
        }

        let mut proof = Vec::with_capacity(N);
        let mut current_index = index;
        let mut layer_index = 0;
        let mut sibling;

        for _ in 0..N {
            if current_index % 2 == 0 {
                sibling = layers[layer_index][current_index + 1];
            } else {
                sibling = layers[layer_index][current_index - 1];
            }

            proof.push(sibling);

            current_index /= 2;
            layer_index += 1;
        }

        proof
    }
}

fn hash_left_right(left: Hash, right: Hash) -> Hash {
    let combined;
    if left.to_bytes() <= right.to_bytes() {
        combined = [b"NODE".as_ref(), left.as_ref(), right.as_ref()];
    } else {
        combined = [b"NODE".as_ref(), right.as_ref(), left.as_ref()];
    }

    hashv(&combined)
}

fn compute_path(proof: &[Hash], leaf: Leaf) -> Vec<Hash> {
    let mut computed_path = Vec::with_capacity(proof.len() + 1);
    let mut computed_hash = Hash::from(leaf);

    computed_path.push(computed_hash);

    for proof_element in proof.iter() {
        computed_hash = hash_left_right(computed_hash, *proof_element);
        computed_path.push(computed_hash);
    }

    computed_path
}

fn is_valid_leaf(proof: &[Hash], root: Hash, leaf: Leaf) -> bool {
    let computed_path = compute_path(proof, leaf);
    is_valid_path(&computed_path, root)
}

fn is_valid_path(path: &[Hash], root: Hash) -> bool {
    if path.is_empty() {
        return false;
    }

    *path.last().unwrap() == root
}

/// Verifies that a given merkle root contains the leaf using the provided proof.
pub fn verify<const N: usize>(root: Hash, proof: &[Hash], leaf: Leaf) -> bool {
    if proof.len() != N {
        return false;
    }

    let computed_path = compute_path(proof, leaf);
    is_valid_path(&computed_path, root)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestTree = MerkleTree<3>;

    #[test]
    fn test_create_tree() {
        let seeds: &[&[u8]] = &[b"test"];
        let tree = TestTree::new(seeds);

        assert_eq!(tree.get_depth(), 3);
        assert_eq!(tree.get_root(), tree.zero_values.last().unwrap().clone());
    }

    #[test]
    fn test_insert_and_remove() {
        let seeds: &[&[u8]] = &[b"test"];

        let mut tree = TestTree::new(seeds);
        let empty = tree.zero_values.first().unwrap().clone();
        let empty_leaf = empty.as_leaf();

        // Tree structure:
        // 
        //              root
        //            /     \
        //         m           n
        //       /   \       /   \
        //      i     j     k     l
        //     / \   / \   / \   / \
        //    a  b  c  d  e  f  g  h

        let a = Hash::from(Leaf::new(&[b"val_1"]));
        let b = Hash::from(Leaf::new(&[b"val_2"]));
        let c = Hash::from(Leaf::new(&[b"val_3"]));

        let d = empty.clone();
        let e = empty.clone();
        let f = empty.clone();
        let g = empty.clone();
        let h = empty.clone();

        let i = hash_left_right(a, b);
        let j: Hash = hash_left_right(c, d);
        let k: Hash = hash_left_right(e, f);
        let l: Hash = hash_left_right(g, h);
        let m: Hash = hash_left_right(i, j);
        let n: Hash = hash_left_right(k, l);
        let root = hash_left_right(m, n);

        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.filled_subtrees[0].eq(&a));

        assert!(tree.try_add(&[b"val_2"]).is_ok());
        assert!(tree.filled_subtrees[0].eq(&a)); // Not a typo

        assert!(tree.try_add(&[b"val_3"]).is_ok());
        assert!(tree.filled_subtrees[0].eq(&c)); // Not a typo

        assert_eq!(tree.filled_subtrees[0], c);
        assert_eq!(tree.filled_subtrees[1], i);
        assert_eq!(tree.filled_subtrees[2], m);
        assert_eq!(root, tree.get_root());

        let val1_proof = vec![b.clone(), j.clone(), n.clone()];
        let val2_proof = vec![a.clone(), j.clone(), n.clone()];
        let val3_proof = vec![d.clone(), i.clone(), n.clone()];

        // Check filled leaves
        assert!(tree.contains(&val1_proof, &[b"val_1"]));
        assert!(tree.contains(&val2_proof, &[b"val_2"]));
        assert!(tree.contains(&val3_proof, &[b"val_3"]));

        // Check empty leaves
        assert!(tree.contains_leaf(&vec![c.clone(), i.clone(), n.clone()], empty_leaf));
        assert!(tree.contains_leaf(&vec![f.clone(), l.clone(), m.clone()], empty_leaf));
        assert!(tree.contains_leaf(&vec![e.clone(), l.clone(), m.clone()], empty_leaf));
        assert!(tree.contains_leaf(&vec![h.clone(), k.clone(), m.clone()], empty_leaf));
        assert!(tree.contains_leaf(&vec![g.clone(), k.clone(), m.clone()], empty_leaf));

        // Remove val2 from the tree
        assert!(tree.try_remove(&val2_proof, &[b"val_2"]).is_ok());

        // Update the expected tree structure
        let i = hash_left_right(a, empty);
        let m: Hash = hash_left_right(i, j);
        let root = hash_left_right(m, n);

        assert_eq!(root, tree.get_root());

        let val1_proof = vec![empty.clone(), j.clone(), n.clone()];
        let val3_proof = vec![d.clone(), i.clone(), n.clone()];

        assert!(tree.contains_leaf(&val1_proof, Leaf::new(&[b"val_1"])));
        assert!(tree.contains_leaf(&val2_proof, empty_leaf));
        assert!(tree.contains_leaf(&val3_proof, Leaf::new(&[b"val_3"])));

        // Check that val2 is no longer in the tree
        assert!(!tree.contains_leaf(&val2_proof, Leaf::new(&[b"val_2"])));

        // Insert val4 into the tree
        assert!(tree.try_add(&[b"val_4"]).is_ok());
        assert!(tree.filled_subtrees[0].eq(&c)); // Not a typo

        // Update the expected tree structure
        let d = Hash::from(Leaf::new(&[b"val_4"]));
        let j = hash_left_right(c, d);
        let m = hash_left_right(i, j);
        let root = hash_left_right(m, n);

        assert_eq!(root, tree.get_root());
    }

    #[test]
    fn test_proof() {
        let seeds: &[&[u8]] = &[b"test"];

        let mut tree = TestTree::new(seeds);

        let leaves = [
            Leaf::new(&[b"val_1"]),
            Leaf::new(&[b"val_2"]),
            Leaf::new(&[b"val_3"]),
        ];

        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.try_add(&[b"val_2"]).is_ok());
        assert!(tree.try_add(&[b"val_3"]).is_ok());

        let val1_proof = tree.get_merkle_proof(&leaves, 0);
        let val2_proof = tree.get_merkle_proof(&leaves, 1);
        let val3_proof = tree.get_merkle_proof(&leaves, 2);

        assert!(tree.contains(&val1_proof, &[b"val_1"]));
        assert!(tree.contains(&val2_proof, &[b"val_2"]));
        assert!(tree.contains(&val3_proof, &[b"val_3"]));

        // Invalid Proof Length
        let invalid_proof_short = &val1_proof[..2]; // Shorter than depth
        let invalid_proof_long = [&val1_proof[..], &val1_proof[..]].concat(); // Longer than depth

        assert!(!tree.contains(&invalid_proof_short, &[b"val_1"]));
        assert!(!tree.contains(&invalid_proof_long, &[b"val_1"]));

        // Empty Proof
        let empty_proof: Vec<Hash> = Vec::new();
        assert!(!tree.contains(&empty_proof, &[b"val_1"]));
    }

    #[test]
    fn test_init_and_reinit() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);
        
        // Store initial state
        let initial_root = tree.get_root();
        let initial_zeros = tree.zero_values;
        let initial_filled = tree.filled_subtrees;
        let initial_index = tree.next_index;

        // Add a leaf to modify the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        
        // Reinitialize
        tree.init(seeds);
        
        // Verify tree is reset to initial state
        assert_eq!(tree.get_root(), initial_root);
        assert_eq!(tree.zero_values, initial_zeros);
        assert_eq!(tree.filled_subtrees, initial_filled);
        assert_eq!(tree.next_index, initial_index);
    }

    #[test]
    fn test_tree_full() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);
        
        // Fill the tree (2^3 = 8 leaves)
        for i in 0u8..8 {
            assert!(tree.try_add(&[&[i]]).is_ok());
        }
        
        // Try to add one more leaf
        let result = tree.try_add(&[b"extra"]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BrineTreeError::TreeFull);
    }

    #[test]
    fn test_replace_leaf() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);
        
        // Add initial leaves
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.try_add(&[b"val_2"]).is_ok());
        
        // Get proof for val_1
        let leaves = [
            Leaf::new(&[b"val_1"]),
            Leaf::new(&[b"val_2"]),
        ];
        let proof = tree.get_merkle_proof(&leaves, 0);
        
        // Replace val_1 with new_val
        assert!(tree.try_replace(&proof, &[b"val_1"], &[b"new_val"]).is_ok());
        
        // Verify new leaf is present
        assert!(tree.contains(&proof, &[b"new_val"]));
        assert!(!tree.contains(&proof, &[b"val_1"]));
        
        // Verify val_2 is still present
        let proof_val2 = tree.get_merkle_proof(&[Leaf::new(&[b"new_val"]), leaves[1]], 1);
        assert!(tree.contains(&proof_val2, &[b"val_2"]));
    }

    #[test]
    fn test_verify() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);
        
        // Add initial leaves
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.try_add(&[b"val_2"]).is_ok());
        
        // Get proof for val_1
        let leaves = [
            Leaf::new(&[b"val_1"]),
            Leaf::new(&[b"val_2"]),
        ];
        let proof = tree.get_merkle_proof(&leaves, 0);
        
        // Verify the proof
        assert!(verify::<3>(tree.get_root(), &proof, Leaf::new(&[b"val_1"])));
    }
}

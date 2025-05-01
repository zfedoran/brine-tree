use bytemuck::{Pod, Zeroable};
use super::{hash::Hash, hash::Leaf, hashv};
use super::utils::check_condition;
use super::error::{BrineTreeError, ProgramResult};

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

    pub fn get_empty_leaf(&self) -> Hash {
        self.zero_values[0]
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

            current_hash = Self::hash_left_right(left, right);
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
        self.try_replace_leaf(proof, Hash::from(leaf), self.get_empty_leaf())
    }

    pub fn try_replace(&mut self, proof: &[Hash], original_data: &[&[u8]], new_data: &[&[u8]]) -> ProgramResult {
        let original_leaf = Leaf::new(original_data);
        let new_leaf = Leaf::new(new_data);
        self.try_replace_leaf(proof, Hash::from(original_leaf), Hash::from(new_leaf))
    }

    pub fn try_replace_leaf(&mut self, proof: &[Hash], original_leaf: Hash, new_leaf: Hash) -> ProgramResult {
        self.check_length(proof)?;

        let original_path = MerkleTree::<N>::compute_path(proof, original_leaf);
        let new_path = MerkleTree::<N>::compute_path(proof, new_leaf);

        check_condition(
            MerkleTree::<N>::is_valid_path(&original_path, self.root),
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
        Self::is_valid_leaf(proof, root, Hash::from(leaf))
    }

    pub fn hash_left_right(left: Hash, right: Hash) -> Hash {
        let combined;
        if left.to_bytes() <= right.to_bytes() {
            combined = [b"NODE".as_ref(), left.as_ref(), right.as_ref()];
        } else {
            combined = [b"NODE".as_ref(), right.as_ref(), left.as_ref()];
        }

        hashv(&combined)
    }

    pub fn compute_path(proof: &[Hash], leaf: Hash) -> Vec<Hash> {
        let mut computed_path = Vec::with_capacity(proof.len() + 1);
        let mut computed_hash = leaf;

        computed_path.push(computed_hash);

        for proof_element in proof.iter() {
            computed_hash = Self::hash_left_right(computed_hash, *proof_element);
            computed_path.push(computed_hash);
        }

        computed_path
    }

    pub fn is_valid_leaf(proof: &[Hash], root: Hash, leaf: Hash) -> bool {
        let computed_path = Self::compute_path(proof, leaf);
        Self::is_valid_path(&computed_path, root)
    }

    pub fn is_valid_path(path: &[Hash], root: Hash) -> bool {
        if path.is_empty() {
            return false;
        }

        *path.last().unwrap() == root
    }

    #[cfg(not(feature = "solana"))]
    fn hash_pairs(pairs: Vec<Hash>) -> Vec<Hash> {
        let mut res = Vec::with_capacity(pairs.len() / 2);

        for i in (0..pairs.len()).step_by(2) {
            let left = pairs[i];
            let right = pairs[i + 1];

            let hashed = Self::hash_left_right(left, right);
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

    fn check_length(&self, proof: &[Hash]) -> Result<(), BrineTreeError> {
        check_condition(proof.len() == N, BrineTreeError::ProofLength)
    }
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

        let i = TestTree::hash_left_right(a, b);
        let j: Hash = TestTree::hash_left_right(c, d);
        let k: Hash = TestTree::hash_left_right(e, f);
        let l: Hash = TestTree::hash_left_right(g, h);
        let m: Hash = TestTree::hash_left_right(i, j);
        let n: Hash = TestTree::hash_left_right(k, l);
        let root = TestTree::hash_left_right(m, n);

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
        let i = TestTree::hash_left_right(a, empty);
        let m: Hash = TestTree::hash_left_right(i, j);
        let root = TestTree::hash_left_right(m, n);

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
        let j = TestTree::hash_left_right(c, d);
        let m = TestTree::hash_left_right(i, j);
        let root = TestTree::hash_left_right(m, n);

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
    fn test_invalid_replace() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);
        
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        
        let leaves = [Leaf::new(&[b"val_1"])];
        let proof = tree.get_merkle_proof(&leaves, 0);
        
        // Try to replace with wrong original leaf
        let result = tree.try_replace(&proof, &[b"wrong_val"], &[b"new_val"]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BrineTreeError::InvalidProof);
        
        // Verify original leaf is still present
        assert!(tree.contains(&proof, &[b"val_1"]));
    }

    #[test]
    fn test_zero_values_calculation() {
        let seeds: &[&[u8]] = &[b"test"];
        let tree = TestTree::new(seeds);
        
        let zeros = tree.zero_values;
        
        // Verify zero values are correctly chained
        let mut expected = hashv(seeds);
        for i in 0..3 {
            assert_eq!(zeros[i], expected);
            expected = hashv(&[b"NODE".as_ref(), expected.as_ref(), expected.as_ref()]);
        }
    }

    #[test]
    fn test_path_computation() {
        let seeds: &[&[u8]] = &[b"test"];
        let tree = TestTree::new(seeds);
        
        let leaf = Hash::from(Leaf::new(&[b"val_1"]));
        let proof = vec![
            Hash::from(Leaf::new(&[b"val_2"])),
            tree.zero_values[1],
            tree.zero_values[2],
        ];
        
        let path = TestTree::compute_path(&proof, leaf);
        
        // Verify path length
        assert_eq!(path.len(), 4);
        
        // Verify path computation
        assert_eq!(path[0], leaf);
        assert_eq!(path[1], TestTree::hash_left_right(leaf, proof[0]));
        assert_eq!(path[2], TestTree::hash_left_right(path[1], proof[1]));
        assert_eq!(path[3], TestTree::hash_left_right(path[2], proof[2]));
    }

    #[test]
    fn test_invalid_path() {
        let seeds: &[&[u8]] = &[b"test"];
        let tree = TestTree::new(seeds);
        
        // Empty path
        let empty_path: Vec<Hash> = vec![];
        assert!(!TestTree::is_valid_path(&empty_path, tree.get_root()));
        
        // Wrong root
        let leaf = Hash::from(Leaf::new(&[b"val_1"]));
        let proof = vec![tree.zero_values[0], tree.zero_values[1], tree.zero_values[2]];
        let path = TestTree::compute_path(&proof, leaf);
        let wrong_root = Hash::default();
        assert!(!TestTree::is_valid_path(&path, wrong_root));
    }

    #[test]
    fn test_hash_left_right_ordering() {
        let left = Hash::from(Leaf::new(&[b"val_1"]));
        let right = Hash::from(Leaf::new(&[b"val_2"]));
        
        let hash1 = TestTree::hash_left_right(left, right);
        
        // Swap order - should produce same result due to ordering in hash_left_right
        let hash2 = TestTree::hash_left_right(right, left);
        
        assert_eq!(hash1, hash2);
        
        // Verify correct ordering was used (smaller hash first)
        let direct = hashv(&[
            b"NODE".as_ref(), 
            left.to_bytes().min(right.to_bytes()).as_ref(), 
            left.to_bytes().max(right.to_bytes()).as_ref()
        ]);

        assert_eq!(hash1, direct);
    }

    #[test]
    fn test_partial_proof() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add a leaf to the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());

        // Create a valid proof
        let leaves = [Leaf::new(&[b"val_1"])];
        let valid_proof = tree.get_merkle_proof(&leaves, 0);
        assert_eq!(valid_proof.len(), 3); // Should match tree depth

        // Create a partial proof (shorter than depth)
        let partial_proof = &valid_proof[..2]; // Take only first two elements

        // Verify that partial proof is rejected
        assert!(!tree.contains(&partial_proof, &[b"val_1"]));

        // Try to use partial proof for removal
        let remove_result = tree.try_remove(&partial_proof, &[b"val_1"]);
        assert!(remove_result.is_err());
        assert_eq!(remove_result.unwrap_err(), BrineTreeError::ProofLength);

        // Try to use partial proof for replacement
        let replace_result = tree.try_replace(&partial_proof, &[b"val_1"], &[b"new_val"]);
        assert!(replace_result.is_err());
        assert_eq!(replace_result.unwrap_err(), BrineTreeError::ProofLength);

        // Verify original leaf is still present
        assert!(tree.contains(&valid_proof, &[b"val_1"]));
    }

    #[test]
    fn test_leaf_vs_node_attack() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add some leaves to the tree
        let val_1 : &[&[u8]] = &[b"val_1"];
        let val_2 : &[&[u8]] = &[b"val_2"];
        assert!(tree.try_add(val_1).is_ok());
        assert!(tree.try_add(val_2).is_ok());

        // Get a valid proof for val_1
        let leaves = [
            Leaf::new(val_1),
            Leaf::new(val_2),
        ];
        let valid_proof = tree.get_merkle_proof(&leaves, 0);
        assert_eq!(valid_proof.len(), 3); // Matches tree depth

        // Create a malicious proof by replacing a node hash with a leaf hash
        let malicious_leaf = Leaf::new(&[b"malicious"]);
        let malicious_hash = Hash::from(malicious_leaf);
        let mut malicious_proof = valid_proof.clone();
        // Replace the second proof element (a node hash) with a leaf hash
        malicious_proof[1] = malicious_hash;

        // Verify that the malicious proof is rejected
        assert!(!tree.contains(&malicious_proof, val_1));

        // Attempt to replace val_1 using the malicious proof
        let replace_result = tree.try_replace(&malicious_proof, val_1, &[b"new_val"]);
        assert!(replace_result.is_err());
        assert_eq!(replace_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Attempt to remove val_1 using the malicious proof
        let remove_result = tree.try_remove(&malicious_proof, val_1);
        assert!(remove_result.is_err());
        assert_eq!(remove_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Verify the tree state is unchanged
        assert!(tree.contains(&valid_proof, val_1));
        assert!(tree.contains(&tree.get_merkle_proof(&leaves, 1), val_2));
    }

    #[test]
    fn test_proof_with_duplicate_hashes() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add a leaf to the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());

        // Create a valid proof
        let leaves = [Leaf::new(&[b"val_1"])];
        let valid_proof = tree.get_merkle_proof(&leaves, 0);
        assert_eq!(valid_proof.len(), 3);

        // Create a proof with duplicate hashes (same hash repeated)
        let duplicate_hash = valid_proof[0];
        let duplicate_proof = vec![duplicate_hash, duplicate_hash, duplicate_hash];

        // Verify that the duplicate proof is rejected
        assert!(!tree.contains(&duplicate_proof, &[b"val_1"]));

        // Attempt to remove using the duplicate proof
        let remove_result = tree.try_remove(&duplicate_proof, &[b"val_1"]);
        assert!(remove_result.is_err());
        assert_eq!(remove_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Attempt to replace using the duplicate proof
        let replace_result = tree.try_replace(&duplicate_proof, &[b"val_1"], &[b"new_val"]);
        assert!(replace_result.is_err());
        assert_eq!(replace_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Verify the original leaf is still present
        assert!(tree.contains(&valid_proof, &[b"val_1"]));
    }

    #[test]
    fn test_proof_with_zero_hashes() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add a leaf to the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());

        // Create a valid proof
        let leaves = [Leaf::new(&[b"val_1"])];
        let valid_proof = tree.get_merkle_proof(&leaves, 0);
        assert_eq!(valid_proof.len(), 3);

        // Create a proof with zero hashes (Hash::default())
        let zero_hash = Hash::default();
        let zero_proof = vec![zero_hash, zero_hash, zero_hash];

        // Verify that the zero proof is rejected
        assert!(!tree.contains(&zero_proof, &[b"val_1"]));

        // Attempt to remove using the zero proof
        let remove_result = tree.try_remove(&zero_proof, &[b"val_1"]);
        assert!(remove_result.is_err());
        assert_eq!(remove_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Attempt to replace using the zero proof
        let replace_result = tree.try_replace(&zero_proof, &[b"val_1"], &[b"new_val"]);
        assert!(replace_result.is_err());
        assert_eq!(replace_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Verify the original leaf is still present
        assert!(tree.contains(&valid_proof, &[b"val_1"]));
    }

    #[test]
    fn test_proof_exploit_hash_ordering() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add two leaves to the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.try_add(&[b"val_2"]).is_ok());

        // Create a valid proof for val_1
        let leaves = [
            Leaf::new(&[b"val_1"]),
            Leaf::new(&[b"val_2"]),
        ];
        let valid_proof = tree.get_merkle_proof(&leaves, 0);
        assert_eq!(valid_proof.len(), 3);

        // Craft a malicious proof by swapping the order of hashes
        // This tests if the hash_left_right ordering (based on byte comparison) can be exploited
        let mut malicious_proof = valid_proof.clone();
        // Swap the first two proof elements to disrupt the expected ordering
        malicious_proof.swap(0, 1);

        // Verify that the malicious proof is rejected
        assert!(!tree.contains(&malicious_proof, &[b"val_1"]));

        // Attempt to remove using the malicious proof
        let remove_result = tree.try_remove(&malicious_proof, &[b"val_1"]);
        assert!(remove_result.is_err());
        assert_eq!(remove_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Attempt to replace using the malicious proof
        let replace_result = tree.try_replace(&malicious_proof, &[b"val_1"], &[b"new_val"]);
        assert!(replace_result.is_err());
        assert_eq!(replace_result.unwrap_err(), BrineTreeError::InvalidProof);

        // Verify the original leaf is still present
        assert!(tree.contains(&valid_proof, &[b"val_1"]));
    }

    // New tests for Empty or Malformed Seeds
    #[test]
    fn test_empty_seeds() {
        let seeds: &[&[u8]] = &[];
        let tree = TestTree::new(seeds);

        // Verify that the tree is initialized correctly with empty seeds
        let zeros = tree.zero_values;
        let mut expected = hashv(seeds); // Should handle empty slice
        for i in 0..3 {
            assert_eq!(zeros[i], expected);
            expected = hashv(&[b"NODE".as_ref(), expected.as_ref(), expected.as_ref()]);
        }

        // Verify that the root and filled subtrees are set correctly
        assert_eq!(tree.get_root(), zeros[2]);
        assert_eq!(tree.filled_subtrees, zeros);
        assert_eq!(tree.next_index, 0);

        // Test adding a leaf to ensure the tree functions
        let mut tree = tree;
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.contains(&tree.get_merkle_proof(&[Leaf::new(&[b"val_1"])], 0), &[b"val_1"]));
    }

    #[test]
    fn test_malformed_seeds_empty_bytes() {
        let seeds: &[&[u8]] = &[b"", b""];
        let tree = TestTree::new(seeds);

        // Verify that the tree is initialized correctly with empty byte arrays
        let zeros = tree.zero_values;
        let mut expected = hashv(seeds); // Should handle empty byte arrays
        for i in 0..3 {
            assert_eq!(zeros[i], expected);
            expected = hashv(&[b"NODE".as_ref(), expected.as_ref(), expected.as_ref()]);
        }

        // Verify that the root and filled subtrees are set correctly
        assert_eq!(tree.get_root(), zeros[2]);
        assert_eq!(tree.filled_subtrees, zeros);
        assert_eq!(tree.next_index, 0);

        // Test adding a leaf to ensure the tree functions
        let mut tree = tree;
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.contains(&tree.get_merkle_proof(&[Leaf::new(&[b"val_1"])], 0), &[b"val_1"]));
    }

    #[test]
    fn test_reinit_with_empty_seeds() {
        let seeds: &[&[u8]] = &[b"test"];
        let mut tree = TestTree::new(seeds);

        // Add a leaf to modify the tree
        assert!(tree.try_add(&[b"val_1"]).is_ok());

        // Reinitialize with empty seeds
        let empty_seeds: &[&[u8]] = &[];
        tree.init(empty_seeds);

        // Verify that the tree is reset correctly
        let zeros = tree.zero_values;
        let mut expected = hashv(empty_seeds);
        for i in 0..3 {
            assert_eq!(zeros[i], expected);
            expected = hashv(&[b"NODE".as_ref(), expected.as_ref(), expected.as_ref()]);
        }

        assert_eq!(tree.get_root(), zeros[2]);
        assert_eq!(tree.filled_subtrees, zeros);
        assert_eq!(tree.next_index, 0);

        // Test adding a leaf after reinitialization
        assert!(tree.try_add(&[b"val_1"]).is_ok());
        assert!(tree.contains(&tree.get_merkle_proof(&[Leaf::new(&[b"val_1"])], 0), &[b"val_1"]));
    }
}

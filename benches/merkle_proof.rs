use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use brine_tree::{ 
    MerkleTree, 
    Leaf,
    Hash, 
    verify, 
    get_merkle_proof, 
    get_split_merkle_proof,
};

const TREE_HEIGHT: usize = 18;
const FILL_COUNT: usize  = 1 << TREE_HEIGHT; // 2^18 = 262,144 leaves

// Select a split layer. Going higher reduces the lower-subtree size.
// Example: layer 10 => lower-subtree width = 2^(18-10) = 256 leaves.
const SPLIT_LAYER: usize = 10;

// Number of random indices to test against the Merkle tree.
const NUM_INDICES: usize = 256;

pub fn merkle_proof_bench(c: &mut Criterion) {
    // Build a reproducible RNG and a set of random leaf indices < FILL_COUNT.
    let mut rng = StdRng::seed_from_u64(42);
    let indices: Vec<usize> = (0..NUM_INDICES).map(|_| rng.gen_range(0..FILL_COUNT)).collect();

    // One-time heavy setup: build the tree and cache the split layer.
    let (tree, leaves_prefix, cached_layer_nodes) = setup_tree_and_layer();

    println!("cached_layer_nodes.len() = {}", cached_layer_nodes.len());

    // Baseline: full `get_merkle_proof` from all known leaves (prefix).
    let mut g = c.benchmark_group("merkle_proof_height_18");
    g.sample_size(10); // keep runtime reasonable for a big tree

    g.bench_function("baseline_full_get_merkle_proof", |b| {
        b.iter_batched(
            || indices[rng.gen_range(0..indices.len())],
            |leaf_index| {
                let proof = get_merkle_proof(
                    &leaves_prefix,
                    &tree.zero_values,
                    black_box(leaf_index),
                    TREE_HEIGHT,
                );
                // Optional: verify to ensure same behavior as split version
                debug_assert!(verify(tree.get_root(), &proof, leaves_prefix[leaf_index]));
                black_box(proof);
            },
            BatchSize::SmallInput,
        )
    });

    // Split: lower-subtree + cached layer nodes
    g.bench_function("split_merkle_proof_cached_layer", |b| {
        // Closure to fetch leaves by global index (sparse DB style).
        let fetch_leaf = |idx: usize| -> Option<Leaf> {
            if idx < FILL_COUNT { Some(leaves_prefix[idx]) } else { None }
        };

        b.iter_batched(
            || indices[rng.gen_range(0..indices.len())],
            |leaf_index| {
                let proof = get_split_merkle_proof(
                    &tree,
                    black_box(leaf_index),
                    SPLIT_LAYER,
                    fetch_leaf,
                    &cached_layer_nodes,
                );
                debug_assert!(verify(tree.get_root(), &proof, leaves_prefix[leaf_index]));
                black_box(proof);
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group!(benches, merkle_proof_bench);
criterion_main!(benches);


fn setup_tree_and_layer() -> (MerkleTree<TREE_HEIGHT>, Vec<Leaf>, Vec<Hash>) {
    // Build the tree with FILL_COUNT leaves. Use compact bytes for Leaf.
    let seeds: &[&[u8]] = &[b"bench"];
    let mut tree = MerkleTree::<TREE_HEIGHT>::new(seeds);

    // Pre-allocate leaves and insert into the tree.
    let mut leaves: Vec<Leaf> = Vec::with_capacity(FILL_COUNT);
    for i in 0..FILL_COUNT {
        // Stable, fast leaf content: little-endian index bytes.
        let bytes = (i as u64).to_le_bytes();
        let leaf = Leaf::new(&[&bytes]);
        leaves.push(leaf);
        // Keep the live tree in sync so `next_index` and `root` are correct.
        tree.try_add_leaf(leaf).expect("tree insert failed");
    }

    // The cached layer is computed from the *contiguous* set of known leaves.
    let cached_layer_nodes = tree.get_layer_nodes(&leaves, SPLIT_LAYER);
    assert!(!cached_layer_nodes.is_empty());

    (tree, leaves, cached_layer_nodes)
}

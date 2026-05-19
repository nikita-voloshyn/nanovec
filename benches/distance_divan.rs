//! Scalar distance baseline (divan).
//!
//! Cross-validates the criterion numbers in `distance.rs` with divan's
//! different statistical approach (median + MAD over a fixed sample budget).

use divan::{black_box, Bencher};
use nanovec::distance::{cosine, dot_product, euclidean};

const DIMS: &[usize] = &[64, 128, 384, 768, 1536];

fn main() {
    divan::main();
}

fn make_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);
    for i in 0..dim {
        a.push((i as f32 * 0.013_37).sin());
        b.push((i as f32 * 0.024_71).cos());
    }
    (a, b)
}

#[divan::bench(args = DIMS)]
fn euclidean_dim(bencher: Bencher, dim: usize) {
    let (a, b) = make_pair(dim);
    bencher.bench_local(|| euclidean(black_box(&a), black_box(&b)));
}

#[divan::bench(args = DIMS)]
fn cosine_dim(bencher: Bencher, dim: usize) {
    let (a, b) = make_pair(dim);
    bencher.bench_local(|| cosine(black_box(&a), black_box(&b)));
}

#[divan::bench(args = DIMS)]
fn dot_product_dim(bencher: Bencher, dim: usize) {
    let (a, b) = make_pair(dim);
    bencher.bench_local(|| dot_product(black_box(&a), black_box(&b)));
}

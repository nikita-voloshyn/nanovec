//! Scalar reference implementations of the distance kernels.
//!
//! These exist for three reasons:
//!  1. Fallback on architectures without a SIMD backend (anything that isn't
//!     `aarch64` or `x86_64`, or x86_64 hosts without AVX2 at runtime).
//!  2. Ground truth for proptests that compare each SIMD backend against the
//!     scalar reference within a tight epsilon.
//!  3. Single source of truth for distance math — the public
//!     `crate::distance::*` wrappers ultimately end up here when no SIMD is
//!     available.

/// Scalar dot product. Caller must ensure `a.len() == b.len()`.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar L2 (Euclidean) distance. Caller must ensure `a.len() == b.len()`.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Scalar cosine distance = 1 - cos_sim. Returns 1.0 if either vector is zero.
/// Caller must ensure `a.len() == b.len()`.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let norm = na.sqrt() * nb.sqrt();
    if norm == 0.0 {
        return 1.0;
    }
    1.0 - dot / norm
}

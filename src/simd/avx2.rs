//! x86_64 AVX2 + FMA implementations of the distance kernels.
//!
//! These functions are unsafe because they require the AVX2 and FMA target
//! features at runtime. The public dispatcher in `super::mod` checks
//! `is_x86_feature_detected!("avx2")` and `"fma"` before calling them.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
};

const LANES: usize = 8;

/// Horizontal sum of an `__m256` to f32.
///
/// # Safety
/// Caller must have AVX (for the 256-bit ops) and SSE3 (for `_mm_hadd_ps`).
/// Both are implied by AVX2 + FMA which the call sites require.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi); // 4-lane sum
    let sum64 = _mm_hadd_ps(sum128, sum128); // pairwise adds
    let sum32 = _mm_hadd_ps(sum64, sum64);
    _mm_cvtss_f32(sum32)
}

/// AVX2 + FMA dot product. Processes 8 f32 lanes per iteration with unroll-by-4
/// (32 floats per loop body), tail loop for partial 8-lane chunks, scalar
/// remainder for the final < 8 elements.
///
/// # Safety
/// Caller must ensure `a.len() == b.len()` AND that the host CPU supports
/// AVX2 and FMA. The dispatcher verifies the latter via
/// `is_x86_feature_detected!`. Pointer math stays in-bounds because `i` is
/// bounded by `unroll_end <= lanes_end <= len`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let unroll = LANES * 4; // 32
    let unroll_end = len - (len % unroll);
    let mut i = 0;
    while i < unroll_end {
        // SAFETY: i + 31 < len because unroll_end is largest multiple of 32 <= len.
        let va0 = _mm256_loadu_ps(pa.add(i));
        let vb0 = _mm256_loadu_ps(pb.add(i));
        let va1 = _mm256_loadu_ps(pa.add(i + 8));
        let vb1 = _mm256_loadu_ps(pb.add(i + 8));
        let va2 = _mm256_loadu_ps(pa.add(i + 16));
        let vb2 = _mm256_loadu_ps(pb.add(i + 16));
        let va3 = _mm256_loadu_ps(pa.add(i + 24));
        let vb3 = _mm256_loadu_ps(pb.add(i + 24));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        i += unroll;
    }

    let lanes_end = len - (len % LANES);
    while i < lanes_end {
        // SAFETY: i + 7 < len.
        let va = _mm256_loadu_ps(pa.add(i));
        let vb = _mm256_loadu_ps(pb.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += LANES;
    }

    let sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum = hsum256(sum_vec);

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// AVX2 + FMA L2 (Euclidean) distance.
///
/// # Safety
/// Same invariants as `dot_avx2`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let unroll = LANES * 4;
    let unroll_end = len - (len % unroll);
    let mut i = 0;
    while i < unroll_end {
        // SAFETY: i + 31 < len.
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
        );
        let d2 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
        );
        let d3 = _mm256_sub_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
        );
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        i += unroll;
    }

    let lanes_end = len - (len % LANES);
    while i < lanes_end {
        // SAFETY: i + 7 < len.
        let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        i += LANES;
    }

    let sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum_sq = hsum256(sum_vec);

    while i < len {
        let d = a[i] - b[i];
        sum_sq += d * d;
        i += 1;
    }
    sum_sq.sqrt()
}

/// AVX2 + FMA cosine distance.
///
/// # Safety
/// Same invariants as `dot_avx2`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut dot_acc = _mm256_setzero_ps();
    let mut na_acc = _mm256_setzero_ps();
    let mut nb_acc = _mm256_setzero_ps();

    let lanes_end = len - (len % LANES);
    let mut i = 0;
    while i < lanes_end {
        // SAFETY: i + 7 < len.
        let va = _mm256_loadu_ps(pa.add(i));
        let vb = _mm256_loadu_ps(pb.add(i));
        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        na_acc = _mm256_fmadd_ps(va, va, na_acc);
        nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
        i += LANES;
    }

    let mut dot = hsum256(dot_acc);
    let mut na = hsum256(na_acc);
    let mut nb = hsum256(nb_acc);

    while i < len {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        i += 1;
    }

    let norm = na.sqrt() * nb.sqrt();
    if norm == 0.0 {
        return 1.0;
    }
    1.0 - dot / norm
}

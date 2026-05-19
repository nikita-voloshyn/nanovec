//! aarch64 NEON implementations of the distance kernels.
//!
//! NEON is part of the base aarch64 ISA, so we don't need a runtime feature
//! check — if we're compiled for aarch64, NEON is guaranteed available.
//! The `#[target_feature(enable = "neon")]` attribute is still required to
//! unlock the intrinsics in stable Rust.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

const LANES: usize = 4;

/// NEON dot product. Processes 4 f32 lanes per iteration with unrolling by 4
/// (16 floats per loop body) to keep the FMA pipeline saturated, then a tail
/// loop, then a scalar remainder.
///
/// # Safety
/// Caller must ensure `a.len() == b.len()`. The aarch64 build target
/// guarantees NEON is available. All pointer arithmetic below stays within
/// the original slice bounds because we only advance `i` by 4 per chunk and
/// stop at the chunk boundary; the scalar tail handles the rest.
#[target_feature(enable = "neon")]
pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    // Four independent accumulators break the dependency chain across FMAs.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let unroll = LANES * 4; // 16
    let unroll_end = len - (len % unroll);
    let mut i = 0;
    while i < unroll_end {
        // SAFETY: `i + 15 < len` because `i < unroll_end` and `unroll_end` is
        // the largest multiple of 16 <= len. Each vld1q_f32 reads 4 lanes.
        let va0 = vld1q_f32(pa.add(i));
        let vb0 = vld1q_f32(pb.add(i));
        let va1 = vld1q_f32(pa.add(i + 4));
        let vb1 = vld1q_f32(pb.add(i + 4));
        let va2 = vld1q_f32(pa.add(i + 8));
        let vb2 = vld1q_f32(pb.add(i + 8));
        let va3 = vld1q_f32(pa.add(i + 12));
        let vb3 = vld1q_f32(pb.add(i + 12));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        acc1 = vfmaq_f32(acc1, va1, vb1);
        acc2 = vfmaq_f32(acc2, va2, vb2);
        acc3 = vfmaq_f32(acc3, va3, vb3);
        i += unroll;
    }

    // 4-lane tail (between unroll_end and the largest multiple of 4 <= len).
    let lanes_end = len - (len % LANES);
    while i < lanes_end {
        // SAFETY: `i + 3 < len` because i < lanes_end and lanes_end is the
        // largest multiple of 4 <= len.
        let va = vld1q_f32(pa.add(i));
        let vb = vld1q_f32(pb.add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += LANES;
    }

    let sum_vec = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut sum = vaddvq_f32(sum_vec);

    // Scalar remainder (0..3 elements).
    while i < len {
        // SAFETY: i < len; in-bounds access via slice indexing.
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// NEON L2 (Euclidean) distance.
///
/// # Safety
/// Same invariants as `dot_neon`: equal-length slices and aarch64 target.
#[target_feature(enable = "neon")]
pub unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let unroll = LANES * 4;
    let unroll_end = len - (len % unroll);
    let mut i = 0;
    while i < unroll_end {
        // SAFETY: i + 15 < len (see dot_neon).
        let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
        let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
        i += unroll;
    }

    let lanes_end = len - (len % LANES);
    while i < lanes_end {
        // SAFETY: i + 3 < len.
        let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        acc0 = vfmaq_f32(acc0, d, d);
        i += LANES;
    }

    let sum_vec = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut sum_sq = vaddvq_f32(sum_vec);

    while i < len {
        let d = a[i] - b[i];
        sum_sq += d * d;
        i += 1;
    }
    sum_sq.sqrt()
}

/// NEON cosine distance.
///
/// # Safety
/// Same invariants as `dot_neon`.
#[target_feature(enable = "neon")]
pub unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    let mut dot_acc = vdupq_n_f32(0.0);
    let mut na_acc = vdupq_n_f32(0.0);
    let mut nb_acc = vdupq_n_f32(0.0);

    let lanes_end = len - (len % LANES);
    let mut i = 0;
    while i < lanes_end {
        // SAFETY: i + 3 < len.
        let va = vld1q_f32(pa.add(i));
        let vb = vld1q_f32(pb.add(i));
        dot_acc = vfmaq_f32(dot_acc, va, vb);
        na_acc = vfmaq_f32(na_acc, va, va);
        nb_acc = vfmaq_f32(nb_acc, vb, vb);
        i += LANES;
    }

    let mut dot = vaddvq_f32(dot_acc);
    let mut na = vaddvq_f32(na_acc);
    let mut nb = vaddvq_f32(nb_acc);

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

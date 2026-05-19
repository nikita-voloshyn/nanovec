//! SIMD dispatch for distance kernels.
//!
//! The public entry points (`dot_product`, `euclidean`, `cosine`) select the
//! best available backend at compile time, with a runtime feature gate on
//! x86_64 to fall back to scalar if AVX2/FMA are unavailable.
//!
//! Backend selection:
//!  - `aarch64` → NEON (always available on aarch64, no runtime check).
//!  - `x86_64`  → AVX2 + FMA if `is_x86_feature_detected!` says so, else scalar.
//!  - anything else → scalar.
//!
//! Every backend has a property-based test that compares it against
//! [`scalar`] within a strict epsilon.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// Dot product of two equal-length f32 slices. Panics if lengths differ.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    dot_product_dispatch(a, b)
}

/// L2 (Euclidean) distance of two equal-length f32 slices. Panics if lengths
/// differ.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    euclidean_dispatch(a, b)
}

/// Cosine distance (= 1 - cosine_similarity) of two equal-length f32 slices.
/// Returns 1.0 if either input has zero norm. Panics if lengths differ.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    cosine_dispatch(a, b)
}

// ---------- per-arch dispatchers ----------
//
// Exactly one of these three is compiled per target. Splitting the dispatch
// into separate cfg-gated functions avoids the clippy `needless_return`
// warning that fires when multiple cfg branches share a function body.

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: aarch64 implies NEON; slice lengths checked by caller.
    unsafe { neon::dot_neon(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn euclidean_dispatch(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: aarch64 implies NEON; slice lengths checked by caller.
    unsafe { neon::euclidean_neon(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: aarch64 implies NEON; slice lengths checked by caller.
    unsafe { neon::cosine_neon(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 + FMA verified at runtime; lengths checked by caller.
        unsafe { avx2::dot_avx2(a, b) }
    } else {
        scalar::dot_product(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn euclidean_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 + FMA verified at runtime; lengths checked by caller.
        unsafe { avx2::euclidean_avx2(a, b) }
    } else {
        scalar::euclidean(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 + FMA verified at runtime; lengths checked by caller.
        unsafe { avx2::cosine_avx2(a, b) }
    } else {
        scalar::cosine(a, b)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    scalar::dot_product(a, b)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn euclidean_dispatch(a: &[f32], b: &[f32]) -> f32 {
    scalar::euclidean(a, b)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn cosine_dispatch(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ---------- Sanity unit tests against known values ----------

    #[test]
    fn dot_product_known_value() {
        assert_relative_eq!(
            dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            32.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn euclidean_known_value() {
        assert_relative_eq!(euclidean(&[0.0, 0.0], &[3.0, 4.0]), 5.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_known_value() {
        // Identical vectors → 0.
        assert_relative_eq!(
            cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            0.0,
            epsilon = 1e-5
        );
        // Orthogonal vectors → 1.
        assert_relative_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_product_remainder_paths() {
        // Hit all the tail paths: length is unroll+lanes+remainder.
        // AVX2 unroll = 32, lanes = 8; NEON unroll = 16, lanes = 4. Choose
        // a length that exercises all three.
        let len = 32 + 8 + 3; // 43
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
        let s = scalar::dot_product(&a, &b);
        let v = dot_product(&a, &b);
        assert_relative_eq!(v, s, epsilon = 1e-3, max_relative = 1e-4);
    }

    #[test]
    fn euclidean_remainder_paths() {
        let len = 32 + 8 + 5;
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.05).cos()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.09).sin()).collect();
        let s = scalar::euclidean(&a, &b);
        let v = euclidean(&a, &b);
        assert_relative_eq!(v, s, epsilon = 1e-3, max_relative = 1e-4);
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        assert_relative_eq!(
            cosine(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]),
            1.0,
            epsilon = 0.0
        );
    }

    // ---------- Property-based equivalence to scalar ----------

    use proptest::prelude::*;

    // Values are bounded to avoid catastrophic cancellation / overflow that
    // makes SIMD vs scalar comparisons fail solely due to summation order.
    // NaN and inf are excluded by construction (finite range).
    fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-100f32..100f32, min_len..=max_len)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn simd_dot_matches_scalar(a in finite_vec(1, 2048)) {
            // Use a single vector against itself + shifted version to get
            // two correlated, finite inputs of equal length.
            let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 0.1).collect();
            prop_assume!(a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()));
            let s = scalar::dot_product(&a, &b);
            let v = dot_product(&a, &b);
            prop_assert!(
                approx::relative_eq!(v, s, epsilon = 1e-3, max_relative = 1e-3),
                "dot SIMD={v} scalar={s} len={}", a.len()
            );
        }

        #[test]
        fn simd_euclidean_matches_scalar(a in finite_vec(1, 2048)) {
            let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 0.1).collect();
            prop_assume!(a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()));
            let s = scalar::euclidean(&a, &b);
            let v = euclidean(&a, &b);
            prop_assert!(
                approx::relative_eq!(v, s, epsilon = 1e-3, max_relative = 1e-3),
                "euclidean SIMD={v} scalar={s} len={}", a.len()
            );
        }

        #[test]
        fn simd_cosine_matches_scalar(a in finite_vec(2, 2048)) {
            // Shift so b is never identical to a but stays correlated.
            let b: Vec<f32> = a.iter().enumerate()
                .map(|(i, x)| x * 0.5 + (i as f32 * 0.01).sin())
                .collect();
            prop_assume!(a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()));
            // Require non-degenerate norms so the divide is well-conditioned.
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(na > 1e-3 && nb > 1e-3);
            let s = scalar::cosine(&a, &b);
            let v = cosine(&a, &b);
            prop_assert!(
                approx::relative_eq!(v, s, epsilon = 1e-3, max_relative = 1e-3),
                "cosine SIMD={v} scalar={s} len={}", a.len()
            );
        }
    }
}

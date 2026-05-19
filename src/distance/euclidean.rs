/// L2 (Euclidean) distance between two equal-length slices.
/// Panics if lengths differ.
///
/// Delegates to the SIMD dispatcher in [`crate::simd`]; on aarch64 this uses
/// NEON, on x86_64 with AVX2+FMA it uses AVX2, otherwise falls back to a
/// scalar reference implementation.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::euclidean(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn euclidean_3_4_triangle() {
        // distance([0,0], [3,4]) == 5.0
        assert_relative_eq!(euclidean(&[0.0, 0.0], &[3.0, 4.0]), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn euclidean_identical_vectors() {
        // distance([1,1,1], [1,1,1]) == 0.0
        assert_relative_eq!(
            euclidean(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn euclidean_single_dimension() {
        // distance([0], [1]) == 1.0
        assert_relative_eq!(euclidean(&[0.0], &[1.0]), 1.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn euclidean_dimension_mismatch_panics() {
        euclidean(&[1.0, 2.0], &[1.0]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn euclidean_non_negative(a in prop::collection::vec(-1000f32..1000f32, 1..=128usize),
                                   b in prop::collection::vec(-1000f32..1000f32, 1..=128usize)) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            prop_assume!(len > 0);
            let d = super::euclidean(a, b);
            prop_assert!(d >= 0.0, "euclidean distance must be non-negative, got {d}");
        }

        #[test]
        fn euclidean_identity(a in prop::collection::vec(-1000f32..1000f32, 1..=128usize)) {
            let d = super::euclidean(&a, &a);
            prop_assert!(d.abs() < 1e-3, "euclidean(a, a) must be ~0, got {d}");
        }

        #[test]
        fn euclidean_symmetric(a in prop::collection::vec(-1000f32..1000f32, 1..=64usize),
                                b in prop::collection::vec(-1000f32..1000f32, 1..=64usize)) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            prop_assume!(len > 0);
            let d_ab = super::euclidean(a, b);
            let d_ba = super::euclidean(b, a);
            prop_assert!((d_ab - d_ba).abs() < 1e-3, "euclidean must be symmetric: {d_ab} vs {d_ba}");
        }
    }
}

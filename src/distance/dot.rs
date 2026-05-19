/// Raw dot product. Higher value = more similar (not a distance metric).
/// Panics if lengths differ.
///
/// Delegates to the SIMD dispatcher in [`crate::simd`]; on aarch64 this uses
/// NEON, on x86_64 with AVX2+FMA it uses AVX2, otherwise falls back to a
/// scalar reference implementation.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_product(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dot_product_basic() {
        // dot_product([1,2,3], [4,5,6]) == 32.0
        assert_relative_eq!(
            dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            32.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn dot_product_orthogonal() {
        // dot_product([1,0], [0,1]) == 0.0
        assert_relative_eq!(dot_product(&[1.0, 0.0], &[0.0, 1.0]), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_simple() {
        // dot_product([2,2], [3,3]) == 12.0
        assert_relative_eq!(dot_product(&[2.0, 2.0], &[3.0, 3.0]), 12.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn dot_product_dimension_mismatch_panics() {
        dot_product(&[1.0, 2.0], &[1.0]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn dot_product_symmetric(a in prop::collection::vec(-100f32..100f32, 1..=64usize),
                                  b in prop::collection::vec(-100f32..100f32, 1..=64usize)) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            prop_assume!(len > 0);
            let d_ab = super::dot_product(a, b);
            let d_ba = super::dot_product(b, a);
            prop_assert!((d_ab - d_ba).abs() < 1e-2, "dot product must be symmetric: {d_ab} vs {d_ba}");
        }

        #[test]
        fn dot_product_zero_vector(a in prop::collection::vec(-100f32..100f32, 1..=64usize)) {
            let zero = vec![0f32; a.len()];
            let d = super::dot_product(&a, &zero);
            prop_assert!(d.abs() < 1e-6, "dot(a, 0) must be 0, got {d}");
        }
    }
}

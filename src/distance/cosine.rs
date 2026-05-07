/// Cosine distance = 1 - cosine_similarity.
/// Returns 0.0 for identical vectors, 1.0 for orthogonal, 2.0 for opposite.
/// Returns 1.0 (max uncertainty) if either vector is zero.
/// Panics if lengths differ.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cosine_identical_vectors() {
        // cosine([1,0,0], [1,0,0]) == 0.0
        assert_relative_eq!(
            cosine(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        // cosine([1,0,0], [0,1,0]) ≈ 1.0
        assert_relative_eq!(
            cosine(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn cosine_opposite_vectors() {
        // cosine([1,0], [-1,0]) ≈ 2.0
        assert_relative_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        assert_relative_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 1.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn cosine_dimension_mismatch_panics() {
        cosine(&[1.0, 2.0], &[1.0]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cosine_range(a in prop::collection::vec(-100f32..100f32, 1..=64usize),
                        b in prop::collection::vec(-100f32..100f32, 1..=64usize)) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            prop_assume!(len > 0);
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(norm_a > 1e-6 && norm_b > 1e-6);
            let d = super::cosine(a, b);
            prop_assert!((-1e-4_f32..=2.0 + 1e-4_f32).contains(&d), "cosine distance out of range [0,2]: {d}");
        }

        #[test]
        fn cosine_identity_is_zero(a in prop::collection::vec(0.01f32..100f32, 1..=64usize)) {
            let d = super::cosine(&a, &a);
            prop_assert!(d.abs() < 1e-4, "cosine(a,a) must be ~0, got {d}");
        }
    }
}

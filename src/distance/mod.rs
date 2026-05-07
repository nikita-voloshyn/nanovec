pub mod cosine;
pub mod dot;
pub mod euclidean;

pub use cosine::cosine;
pub use dot::dot_product;
pub use euclidean::euclidean;

/// Distance metric selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

/// Trait for distance computation.
pub trait Distance: Send + Sync {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

struct EuclideanDist;
struct CosineDist;
struct DotProductDist;

impl Distance for EuclideanDist {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        euclidean(a, b)
    }
}

impl Distance for CosineDist {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine(a, b)
    }
}

impl Distance for DotProductDist {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        -dot_product(a, b)
    }
}

/// Returns a boxed distance function for the given metric.
pub fn distance_fn(metric: Metric) -> Box<dyn Distance> {
    match metric {
        Metric::Euclidean => Box::new(EuclideanDist),
        Metric::Cosine => Box::new(CosineDist),
        Metric::DotProduct => Box::new(DotProductDist),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn distance_fn_euclidean() {
        let d = distance_fn(Metric::Euclidean);
        assert_relative_eq!(d.compute(&[0.0, 0.0], &[3.0, 4.0]), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn distance_fn_dot_product() {
        let d = distance_fn(Metric::DotProduct);
        // Negated so higher similarity = lower score (fits min-distance heap).
        assert_relative_eq!(
            d.compute(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            -32.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn distance_fn_cosine() {
        let d = distance_fn(Metric::Cosine);
        assert_relative_eq!(
            d.compute(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]),
            0.0,
            epsilon = 1e-6
        );
    }
}

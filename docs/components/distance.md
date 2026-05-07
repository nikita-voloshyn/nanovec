# Distance Functions

## Purpose

The distance module provides scalar implementations of three vector similarity/distance functions, a runtime-dispatch abstraction (`Distance` trait), and a factory function (`distance_fn`) for selecting among them. All functions are pure: no global state, no allocation, no unsafe code.

## Public API

```rust
// src/distance/mod.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

pub trait Distance: Send + Sync {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

pub fn distance_fn(metric: Metric) -> Box<dyn Distance>;

// Re-exported free functions:
pub fn euclidean(a: &[f32], b: &[f32]) -> f32;   // src/distance/euclidean.rs
pub fn cosine(a: &[f32], b: &[f32]) -> f32;       // src/distance/cosine.rs
pub fn dot_product(a: &[f32], b: &[f32]) -> f32;  // src/distance/dot.rs
```

All three free functions panic if `a.len() != b.len()`.

## Distance Functions

### Euclidean (L2 distance)

**Formula:** `sqrt( sum( (a_i - b_i)^2 ) )`

**Range:** [0, +inf)

**Interpretation:** 0 means identical vectors; higher values mean greater spatial distance.

**When to use:** General-purpose similarity when vectors live in a meaningful geometric space and are not normalized. Magnitude differences matter.

```rust
use nanovec::distance::euclidean;

let d = euclidean(&[0.0, 0.0], &[3.0, 4.0]);
assert!((d - 5.0).abs() < 1e-6); // 3-4-5 right triangle
```

### Cosine distance

**Formula:** `1 - dot(a, b) / (|a| * |b|)`

**Range:** [0, 2]

| Result | Meaning |
|--------|---------|
| 0.0 | Identical direction |
| 1.0 | Orthogonal (or either vector is zero) |
| 2.0 | Opposite direction |

**Zero-vector guard:** If either input has zero norm, the function returns 1.0 (maximum uncertainty) rather than dividing by zero.

**When to use:** Text embeddings and any setting where only the direction of the vector matters, not its magnitude. Equivalent to Euclidean distance on unit-normalized vectors.

```rust
use nanovec::distance::cosine;

assert!((cosine(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0])).abs() < 1e-6); // 0.0
assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);     // 1.0 (orthogonal)
assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);    // 2.0 (opposite)
```

### Dot product

**Formula:** `sum( a_i * b_i )`

**Range:** (-inf, +inf) -- unbounded, not a true distance metric

**Interpretation:** Higher values indicate greater similarity. Unlike Euclidean and cosine, dot product is not a distance; a smaller return value does not mean "closer."

**When to use:** When vectors are pre-normalized to unit length (in which case dot product equals cosine similarity) or when the search engine that produced the embeddings is trained with dot-product objectives (e.g., DPR, some OpenAI embeddings).

```rust
use nanovec::distance::dot_product;

let d = dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
assert!((d - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
```

## Runtime Dispatch

`distance_fn(metric)` returns a `Box<dyn Distance>` that can be stored and called uniformly without knowing the underlying metric at compile time. This is used by `BruteForce::search`.

```rust
use nanovec::distance::{distance_fn, Metric};

let dist = distance_fn(Metric::Euclidean);
let score = dist.compute(&[1.0, 0.0], &[0.0, 1.0]);
```

The `Distance` trait is `Send + Sync`, so the boxed object can cross thread boundaries if needed in the future.

## Metric Selection Guide

| Scenario | Recommended metric |
|----------|--------------------|
| Raw geometric space, magnitudes meaningful | `Euclidean` |
| Text/language model embeddings, not normalized | `Cosine` |
| Embeddings pre-normalized to unit length | `DotProduct` (equivalent to cosine, faster) |
| Maximum inner product search (MIPS) | `DotProduct` |

## Performance Characteristics

All three functions are O(d) where d is the vector dimension. They use iterator chains that compile to tight scalar loops. No SIMD acceleration is present in Phase 1; that is planned for Phase 2 (`src/simd/`).

## Dependencies

- No other NanoVec modules.
- `approx` crate used in tests only (not a runtime dependency).

## Test Coverage

16 unit tests and 7 proptest properties across `src/distance/euclidean.rs`, `src/distance/cosine.rs`, `src/distance/dot.rs`, and `src/distance/mod.rs`.

### Unit tests (by file)

**euclidean.rs:** 3-4-5 triangle, identical vectors, single dimension, dimension-mismatch panic.

**cosine.rs:** Identical vectors (0.0), orthogonal (1.0), opposite (2.0), zero vector (1.0), dimension-mismatch panic.

**dot.rs:** Basic product, orthogonal vectors (0.0), simple product, dimension-mismatch panic.

**mod.rs:** `distance_fn` dispatch for all three metrics.

### Proptest properties

| Property | Invariant |
|----------|-----------|
| `euclidean_non_negative` | `euclidean(a, b) >= 0` for all inputs |
| `euclidean_identity` | `euclidean(a, a) ~= 0` |
| `euclidean_symmetric` | `euclidean(a, b) ~= euclidean(b, a)` |
| `cosine_range` | `cosine(a, b)` in `[0, 2]` for non-zero vectors |
| `cosine_identity_is_zero` | `cosine(a, a) ~= 0` for positive vectors |
| `dot_product_symmetric` | `dot(a, b) ~= dot(b, a)` |
| `dot_product_zero_vector` | `dot(a, zero) ~= 0` |

Tolerance used in properties: `1e-3` for euclidean symmetry/identity, `1e-4` for cosine, `1e-2` for dot symmetry. These accommodate float accumulation error over up to 128-element vectors.

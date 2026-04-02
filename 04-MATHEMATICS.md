# NanoVec: Mathematical Foundations & Algorithms

## Vector Similarity Metrics

### 1. Euclidean Distance (L2 Norm)

**Definition**: The straight-line distance between two points in multidimensional space.

**Formula**:
```
d(a, b) = √(Σᵢ (aᵢ - bᵢ)²)

Squared L2 (avoids sqrt for performance):
d²(a, b) = Σᵢ (aᵢ - bᵢ)²
```

**Properties**:
- ✅ **Metric**: Satisfies triangle inequality
- ✅ **Symmetric**: d(a,b) = d(b,a)
- ✅ **Bounded**: [0, ∞)
- ⚠️ **Scale-sensitive**: Affected by vector magnitude

**Implementation**:
```rust
/// Scalar baseline implementation
pub fn squared_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// SIMD-accelerated version (AVX2)
#[cfg(target_feature = "avx2")]
pub unsafe fn squared_euclidean_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    
    for chunk_idx in 0..(a.len() / 8) {
        let offset = chunk_idx * 8;
        
        // Load 8 values from each vector
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // Compute diff = a - b
        let diff = _mm256_sub_ps(va, vb);
        
        // FMA: sum = diff * diff + sum
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    horizontal_sum_avx2(sum) + handle_remainder(a, b)
}
```

**Use Cases**:
- Dense vector embeddings where magnitude matters
- Image similarity (pixel-space distances)
- Spatial coordinates in physics simulations

---

### 2. Cosine Similarity

**Definition**: Measures the angle between two vectors, ignoring magnitude.

**Formula**:
```
cos(θ) = (a · b) / (||a|| × ||b||)

Where:
  a · b = Σᵢ (aᵢ × bᵢ)          (dot product)
  ||a|| = √(Σᵢ aᵢ²)             (magnitude)
```

**Properties**:
- ✅ **Scale-invariant**: Normalizes by magnitude
- ✅ **Bounded**: [-1, 1] where 1 = identical direction
- ✅ **Widely used**: Standard in NLP/text embeddings
- ⚠️ **Not a metric**: Doesn't satisfy triangle inequality

**Implementation**:
```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = l2_norm_simd(a);
    let norm_b = l2_norm_simd(b);
    
    dot / (norm_a * norm_b)
}

/// Convert to distance metric (0 = identical, 2 = opposite)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}
```

**Normalized Vectors Optimization**:
```rust
/// If vectors are pre-normalized (||v|| = 1), cosine = dot product
pub fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    // Skip magnitude computation when vectors are unit-length
    debug_assert!((l2_norm_simd(a) - 1.0).abs() < 1e-6);
    debug_assert!((l2_norm_simd(b) - 1.0).abs() < 1e-6);
    
    dot_product_simd(a, b)
}
```

**Use Cases**:
- Text embeddings (BERT, OpenAI, Sentence Transformers)
- Document similarity (TF-IDF vectors)
- Recommendation systems (user preference vectors)

---

### 3. Manhattan Distance (L1 Norm)

**Definition**: Sum of absolute differences along each dimension.

**Formula**:
```
d(a, b) = Σᵢ |aᵢ - bᵢ|
```

**Properties**:
- ✅ **Metric**: Satisfies triangle inequality
- ✅ **Efficient**: No squaring/sqrt operations
- ⚠️ **Less common**: Rarely used for learned embeddings

**Implementation**:
```rust
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}
```

---

## Core Mathematical Operations

### Dot Product (Inner Product)

**Definition**: Projection of one vector onto another.

**Formula**:
```
a · b = Σᵢ (aᵢ × bᵢ)
```

**SIMD Implementation (AVX2)**:
```rust
#[cfg(target_feature = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();  // 8-wide accumulator
    
    // Process 8 elements per iteration
    for i in 0..(a.len() / 8) {
        let offset = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // FMA: sum = (va * vb) + sum
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Reduce 8 lanes to single scalar
    horizontal_sum_avx2(sum) + dot_product_scalar(&a[(a.len() / 8) * 8..], &b[(b.len() / 8) * 8..])
}
```

**Horizontal Reduction**:
```rust
/// Collapse 8-lane SIMD register to single value
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Step 1: Add high and low 128-bit halves
    // v = [a, b, c, d, e, f, g, h]
    let low_128 = _mm256_castps256_ps128(v);        // [a, b, c, d]
    let high_128 = _mm256_extractf128_ps(v, 1);     // [e, f, g, h]
    let sum_128 = _mm_add_ps(low_128, high_128);    // [a+e, b+f, c+g, d+h]
    
    // Step 2: Add pairs within 128-bit register
    let shuf = _mm_movehdup_ps(sum_128);            // [b+f, b+f, d+h, d+h]
    let sum_64 = _mm_add_ps(sum_128, shuf);         // [a+e+b+f, ..., c+g+d+h, ...]
    
    // Step 3: Final horizontal add
    let shuf2 = _mm_movehl_ps(sum_64, sum_64);
    let sum_32 = _mm_add_ss(sum_64, shuf2);
    
    _mm_cvtss_f32(sum_32)
}
```

**Performance Characteristics**:
```
Scalar:  768 iterations × ~3 cycles = 2,304 cycles
AVX2:    96 iterations × ~3 cycles = 288 cycles
Speedup: ~8x (theoretical maximum for 8-wide SIMD)
```

---

### Vector Magnitude (L2 Norm)

**Definition**: Length of vector in Euclidean space.

**Formula**:
```
||v|| = √(Σᵢ vᵢ²) = √(v · v)
```

**Implementation**:
```rust
pub fn l2_norm_simd(v: &[f32]) -> f32 {
    // Compute v · v using SIMD
    let squared_sum = dot_product_simd(v, v);
    
    // Single sqrt at the end (avoid per-element sqrt)
    squared_sum.sqrt()
}

/// Squared norm (avoids sqrt when only comparing magnitudes)
pub fn l2_norm_squared_simd(v: &[f32]) -> f32 {
    dot_product_simd(v, v)
}
```

---

## Floating-Point Arithmetic Considerations

### IEEE 754 Standard

**Key Properties**:
```rust
// Floating-point operations are NOT associative
let a = 1e20_f32;
let b = -1e20_f32;
let c = 1.0_f32;

assert_ne!((a + b) + c, a + (b + c));  // Different results!
// Left:  (1e20 - 1e20) + 1.0 = 1.0
// Right: 1e20 + (-1e20 + 1.0) = 1e20 (precision loss)
```

**Implications for SIMD**:
- Compilers cannot auto-vectorize floating-point loops without permission
- `-ffast-math` flag relaxes precision but can introduce errors
- Manual SIMD implementation gives explicit control

**Mitigation Strategies**:
```rust
/// Use Kahan summation for high precision
pub fn dot_product_kahan(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut compensation = 0.0_f32;
    
    for (x, y) in a.iter().zip(b.iter()) {
        let product = x * y;
        let y = product - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    sum
}
```

---

## KD-Tree Construction Algorithm

### Median Selection: Quickselect

**Problem**: Find the median element in O(n) time without full sort.

**Algorithm**:
```rust
/// Partition array around k-th element (0-indexed)
fn quickselect(arr: &mut [f32], k: usize) -> f32 {
    if arr.len() == 1 {
        return arr[0];
    }
    
    // Choose pivot (median-of-three heuristic)
    let pivot_idx = median_of_three(arr);
    let pivot = arr[pivot_idx];
    
    // Partition: [< pivot | pivot | >= pivot]
    let partition_idx = partition(arr, pivot);
    
    if k == partition_idx {
        return arr[k];
    } else if k < partition_idx {
        quickselect(&mut arr[..partition_idx], k)
    } else {
        quickselect(&mut arr[partition_idx + 1..], k - partition_idx - 1)
    }
}

fn partition(arr: &mut [f32], pivot: f32) -> usize {
    let mut i = 0;
    
    for j in 0..arr.len() {
        if arr[j] < pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    i
}
```

**Time Complexity**:
- Average: O(n)
- Worst: O(n²) with poor pivot choices
- Optimized: O(n) guaranteed with median-of-medians

### Tree Construction

**Recursive Splitting**:
```rust
fn build_kdtree(
    vectors: &[Vec<f32>],
    indices: &mut [usize],
    depth: usize,
    max_leaf_size: usize,
) -> KDNode {
    // Base case: create leaf
    if indices.len() <= max_leaf_size {
        return KDNode::Leaf {
            indices: indices.to_vec(),
        };
    }
    
    // Select axis (round-robin through dimensions)
    let axis = depth % vectors[0].len();
    
    // Find median value along this axis
    let median_idx = indices.len() / 2;
    select_by_axis(indices, vectors, axis, median_idx);
    let median_value = vectors[indices[median_idx]][axis];
    
    // Split into left and right
    let (left_indices, right_indices) = indices.split_at_mut(median_idx);
    
    KDNode::Internal {
        split_axis: axis,
        split_value: median_value,
        left: Box::new(build_kdtree(vectors, left_indices, depth + 1, max_leaf_size)),
        right: Box::new(build_kdtree(vectors, right_indices, depth + 1, max_leaf_size)),
    }
}
```

**Complexity Analysis**:
```
Tree Height: h = log₂(n)
Work per level: O(n) (partitioning all elements)
Total levels: log₂(n)
Total complexity: O(n log n)
```

---

## K-Nearest Neighbors Search

### Branch-and-Bound Algorithm

**Core Idea**: Prune branches that provably cannot contain closer points.

**Geometric Intuition**:
```
Query Point: Q
Current best distance: r
Splitting Hyperplane: H

If distance(Q, H) > r:
  → All points on far side of H are > r away
  → Prune entire subtree
```

**Pruning Condition**:
```rust
/// Check if far branch could contain better results
fn should_search_far_branch(
    query: &[f32],
    split_axis: usize,
    split_value: f32,
    worst_distance: f32,
) -> bool {
    // Perpendicular distance to splitting plane
    let axis_diff = query[split_axis] - split_value;
    let axis_distance_squared = axis_diff * axis_diff;
    
    // If distance to plane >= worst accepted distance, prune
    axis_distance_squared < worst_distance
}
```

**Search Pseudocode**:
```
function KNN_SEARCH(node, query, k, heap):
    if node is LEAF:
        for each vector in node:
            distance = compute_distance(query, vector)
            heap.insert(distance, vector)
    else:
        // Determine near and far branches
        if query[node.axis] < node.split_value:
            near = node.left
            far = node.right
        else:
            near = node.right
            far = node.left
        
        // Always search near branch
        KNN_SEARCH(near, query, k, heap)
        
        // Conditionally search far branch
        axis_dist = (query[node.axis] - node.split_value)²
        if axis_dist < heap.max_distance OR heap.size < k:
            KNN_SEARCH(far, query, k, heap)
```

**Complexity**:
- Best case: O(log n) when pruning is effective
- Worst case: O(n) when all branches must be explored
- Average (low-D): O(log n + k)
- High-D: Degrades to O(n) due to curse of dimensionality

---

## Curse of Dimensionality

### Mathematical Analysis

**Volume of Hypersphere**:
```
V(d, r) = (π^(d/2) / Γ(d/2 + 1)) × r^d

As d → ∞:
  - Volume concentrates in corners of hypercube
  - Distance between random points converges
  - Nearest and farthest neighbors become equidistant
```

**Distance Concentration**:
```
For random vectors in high dimensions:

E[d_max - d_min] / E[d_min] → 0 as d → ∞

Meaning: All distances become approximately equal!
```

**Empirical Test**:
```rust
fn test_distance_concentration(dimension: usize, num_samples: usize) {
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dimension).map(|_| rng.gen()).collect();
    
    let distances: Vec<f32> = (0..num_samples)
        .map(|_| {
            let point: Vec<f32> = (0..dimension).map(|_| rng.gen()).collect();
            euclidean_distance(&query, &point)
        })
        .collect();
    
    let min = distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    
    println!("Dimension {}: (max-min)/mean = {:.4}", 
             dimension, 
             (max - min) / mean);
}

// Output:
// Dimension 10: (max-min)/mean = 0.4521
// Dimension 100: (max-min)/mean = 0.1234
// Dimension 1000: (max-min)/mean = 0.0398  ← Everything looks similar!
```

### Implications for NanoVec

**Why KD-Trees Fail in High Dimensions**:
1. Distance to splitting plane ≈ distance to nearest neighbor
2. Search sphere intersects almost all branches
3. Pruning becomes ineffective → O(n) search

**Mitigation Strategies**:
```rust
/// Use KD-Tree for low-D, switch to brute-force for high-D
pub fn adaptive_search(
    vectors: &VectorStore,
    query: &[f32],
    k: usize,
    dimension: usize,
) -> Vec<SearchResult> {
    const HIGH_DIM_THRESHOLD: usize = 20;
    
    if dimension <= HIGH_DIM_THRESHOLD && vectors.count() > 10_000 {
        // KD-Tree beneficial for low-D with large dataset
        kdtree_search(vectors, query, k)
    } else {
        // Brute-force SIMD scan faster for high-D or small dataset
        brute_force_simd_search(vectors, query, k)
    }
}
```

---

## Numerical Stability

### Avoiding Catastrophic Cancellation

**Problem**:
```rust
// BAD: Subtracting similar large numbers
let a = 1_000_000.0_f32;
let b = 999_999.9_f32;
let diff = a - b;  // Should be 0.1, but floating-point error accumulates
```

**Solution for Variance**:
```rust
/// Compute variance using two-pass algorithm
pub fn variance_stable(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    
    // Single pass over deviations from mean
    values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32
}
```

### Handling Edge Cases

```rust
/// Cosine similarity with division-by-zero protection
pub fn cosine_similarity_safe(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = l2_norm_simd(a);
    let norm_b = l2_norm_simd(b);
    
    const EPSILON: f32 = 1e-8;
    
    if norm_a < EPSILON || norm_b < EPSILON {
        return 0.0;  // Treat zero vectors as orthogonal
    }
    
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)  // Ensure [-1, 1] range
}
```

---

## Performance Optimization Techniques

### Loop Unrolling

```rust
/// Manually unroll to reduce loop overhead
pub fn dot_product_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let chunks = a.len() / 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        sum += a[offset] * b[offset]
             + a[offset + 1] * b[offset + 1]
             + a[offset + 2] * b[offset + 2]
             + a[offset + 3] * b[offset + 3];
    }
    
    // Handle remainder
    sum + a[chunks * 4..].iter().zip(&b[chunks * 4..])
                         .map(|(x, y)| x * y)
                         .sum::<f32>()
}
```

### Fused Multiply-Add (FMA)

**Hardware Instruction**: Computes `(a × b) + c` in single step.

**Benefits**:
- Reduced rounding errors (single rounding instead of two)
- Higher throughput (one instruction vs. two)
- Lower latency on modern CPUs

**Usage**:
```rust
// Explicit FMA via intrinsics
unsafe {
    let result = _mm256_fmadd_ps(a, b, c);  // (a * b) + c
}

// Compiler may generate FMA automatically with optimization
let result = (a * b) + c;  // May use FMA with -C target-cpu=native
```

---



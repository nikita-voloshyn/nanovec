# NanoVec: Detailed Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent / LLM Client                    │
│                  (Claude, LangGraph, etc.)                  │
└────────────────────┬────────────────────────────────────────┘
                     │ MCP JSON-RPC
                     │ (stdio/SSE)
┌────────────────────▼────────────────────────────────────────┐
│                   MCP Server Layer                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Tool: index_document(text, metadata)                │  │
│  │  Tool: semantic_search(query, k)                     │  │
│  │  Tool: delete_document(id)                           │  │
│  │  Tool: get_stats()                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Function Calls
┌────────────────────▼────────────────────────────────────────┐
│              Vector Database Core Engine                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Query Orchestrator                         │  │
│  │  • Request routing                                   │  │
│  │  • Result aggregation                                │  │
│  │  • Error handling                                    │  │
│  └──────────┬────────────────────────┬──────────────────┘  │
│             │                        │                      │
│  ┌──────────▼──────────┐  ┌─────────▼──────────────────┐  │
│  │  Indexing Pipeline  │  │  Search Engine             │  │
│  │                     │  │                            │  │
│  │  • Text chunking    │  │  • Query embedding         │  │
│  │  • Embedding call   │  │  • Index traversal         │  │
│  │  • Vector insert    │  │  • Distance computation    │  │
│  │  • Index rebuild    │  │  • Result ranking          │  │
│  └─────────────────────┘  └────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Memory Management Layer                   │  │
│  │  ┌───────────────────┐  ┌──────────────────────┐    │  │
│  │  │  Vector Storage   │  │  Metadata Storage    │    │  │
│  │  │                   │  │                      │    │  │
│  │  │  Flat f32 array   │  │  VectorRecord[]      │    │  │
│  │  │  (SoA layout)     │  │  (ID, offset, meta)  │    │  │
│  │  └───────────────────┘  └──────────────────────┘    │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │         Spatial Index (KD-Tree)               │  │  │
│  │  │  • Hierarchical partitioning                  │  │  │
│  │  │  • Balanced binary tree                       │  │  │
│  │  │  • Leaf nodes → Vector offsets                │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        SIMD Computation Layer                        │  │
│  │  • Dot product (FMA instructions)                    │  │
│  │  • Cosine similarity                                 │  │
│  │  • Euclidean distance (L2)                           │  │
│  │  • Horizontal reduction                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### 1. Vector Storage (Structure of Arrays)

```rust
/// The heart of NanoVec: contiguous memory for maximum cache efficiency
pub struct VectorStore {
    /// Flat array of f32 values: [v1_d1, v1_d2, ..., v1_dN, v2_d1, v2_d2, ...]
    /// Guaranteed contiguous allocation for spatial locality
    vectors: Vec<f32>,
    
    /// Dimensionality of each vector (e.g., 768 for many embedding models)
    dimension: usize,
    
    /// Total number of vectors stored (vectors.len() / dimension)
    count: usize,
}

impl VectorStore {
    /// Insert a new vector, returns its index
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dimension);
        
        let index = self.count;
        self.vectors.extend(vector);
        self.count += 1;
        
        index
    }
    
    /// Get a slice view of a specific vector (zero-copy)
    pub fn get(&self, index: usize) -> &[f32] {
        let start = index * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
    
    /// Iterate over all vectors as slices
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.vectors.chunks_exact(self.dimension)
    }
}
```

**Memory Layout**:
```
Cache Line 1 (64 bytes): [v1_d0, v1_d1, ..., v1_d15]  ← 16 f32 values
Cache Line 2 (64 bytes): [v1_d16, v1_d17, ..., v1_d31]
...
Cache Line N (64 bytes): [v2_d0, v2_d1, ..., v2_d15]
```

### 2. Metadata Records (Decoupled Storage)

```rust
/// Lightweight record linking IDs to vector offsets
pub struct VectorRecord {
    /// Unique document identifier (user-provided or auto-generated)
    pub id: u64,
    
    /// Index into VectorStore (NOT byte offset)
    pub vector_index: usize,
    
    /// Arbitrary metadata (JSON, text, etc.)
    pub metadata: String,
    
    /// Optional timestamp for temporal queries
    pub timestamp: Option<u64>,
}

/// Collection of all records
pub struct RecordStore {
    records: Vec<VectorRecord>,
    
    /// HashMap for O(1) ID → record index lookup
    id_to_index: HashMap<u64, usize>,
}
```

**Design Rationale**:
- **Separation of Concerns**: Math operations scan only `VectorStore`
- **Cache Efficiency**: Avoid polluting cache lines with string metadata during distance calculations
- **Flexibility**: Metadata can be arbitrarily large without affecting vector operations

### 3. Spatial Index: KD-Tree

```rust
/// K-Dimensional Tree node (recursive structure)
pub enum KDNode {
    /// Internal node: splits space along one dimension
    Internal {
        /// Which dimension to split on (0 to dimension-1, cycling)
        split_axis: usize,
        
        /// Median value along split_axis
        split_value: f32,
        
        /// Left subtree (values < split_value)
        left: Box<KDNode>,
        
        /// Right subtree (values >= split_value)
        right: Box<KDNode>,
    },
    
    /// Leaf node: contains actual vector indices
    Leaf {
        /// Vector indices stored in this leaf
        indices: Vec<usize>,
    },
}

pub struct KDTree {
    root: Option<Box<KDNode>>,
    dimension: usize,
    
    /// Maximum vectors per leaf before splitting
    max_leaf_size: usize,
}
```

**Construction Algorithm**:
```rust
impl KDTree {
    /// Build tree from vector indices (O(n log n) time)
    pub fn build(
        &mut self, 
        vectors: &VectorStore, 
        indices: Vec<usize>
    ) {
        self.root = Some(self.build_recursive(
            vectors, 
            indices, 
            0  // Start with axis 0
        ));
    }
    
    fn build_recursive(
        &self,
        vectors: &VectorStore,
        mut indices: Vec<usize>,
        depth: usize
    ) -> Box<KDNode> {
        // Base case: create leaf if below threshold
        if indices.len() <= self.max_leaf_size {
            return Box::new(KDNode::Leaf { indices });
        }
        
        // Select splitting axis (cycle through dimensions)
        let axis = depth % self.dimension;
        
        // Find median along this axis using quickselect O(n)
        let median_idx = indices.len() / 2;
        let (_, median_val, _) = select_median_by_axis(
            &mut indices, 
            vectors, 
            axis, 
            median_idx
        );
        
        // Partition into left/right based on median
        let (left_indices, right_indices) = partition_by_value(
            indices, 
            vectors, 
            axis, 
            median_val
        );
        
        // Recursively build subtrees
        Box::new(KDNode::Internal {
            split_axis: axis,
            split_value: median_val,
            left: self.build_recursive(vectors, left_indices, depth + 1),
            right: self.build_recursive(vectors, right_indices, depth + 1),
        })
    }
}
```

### 4. Priority Queue: Max-Heap

```rust
/// Fixed-size max-heap for top-K results
pub struct BoundedMaxHeap {
    /// Heap storage: [distance, vector_index]
    heap: Vec<(f32, usize)>,
    
    /// Maximum capacity (K in KNN search)
    capacity: usize,
}

impl BoundedMaxHeap {
    /// Create new heap with fixed capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Insert element, maintaining heap property
    pub fn push(&mut self, distance: f32, index: usize) {
        if self.heap.len() < self.capacity {
            // Heap not full: add and bubble up
            self.heap.push((distance, index));
            self.sift_up(self.heap.len() - 1);
        } else if distance < self.heap[0].0 {
            // Better than worst: replace root and sift down
            self.heap[0] = (distance, index);
            self.sift_down(0);
        }
    }
    
    /// Get worst distance (root of max-heap)
    pub fn worst_distance(&self) -> Option<f32> {
        self.heap.first().map(|(dist, _)| *dist)
    }
    
    /// Bubble element up to restore heap property
    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            if self.heap[idx].0 > self.heap[parent_idx].0 {
                self.heap.swap(idx, parent_idx);
                idx = parent_idx;
            } else {
                break;
            }
        }
    }
    
    /// Bubble element down to restore heap property
    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left_child = 2 * idx + 1;
            let right_child = 2 * idx + 2;
            let mut largest = idx;
            
            if left_child < self.heap.len() 
                && self.heap[left_child].0 > self.heap[largest].0 {
                largest = left_child;
            }
            
            if right_child < self.heap.len() 
                && self.heap[right_child].0 > self.heap[largest].0 {
                largest = right_child;
            }
            
            if largest != idx {
                self.heap.swap(idx, largest);
                idx = largest;
            } else {
                break;
            }
        }
    }
    
    /// Extract sorted results (converts to min-heap order)
    pub fn into_sorted_results(mut self) -> Vec<(f32, usize)> {
        let mut results = Vec::with_capacity(self.heap.len());
        
        while !self.heap.is_empty() {
            // Pop root (max element)
            let root = self.heap.swap_remove(0);
            if !self.heap.is_empty() {
                self.sift_down(0);
            }
            results.push(root);
        }
        
        results.reverse();  // Convert max-heap order to ascending
        results
    }
}
```

**Heap Indexing**:
```
Array: [60, 45, 50, 30, 40, 35, 25]
         0   1   2   3   4   5   6

Tree Structure:
          60 (idx=0)
        /            \
      45 (idx=1)    50 (idx=2)
     /    \         /    \
   30(3)  40(4)   35(5)  25(6)

Parent of idx: (idx-1)/2
Left child: 2*idx + 1
Right child: 2*idx + 2
```

## Query Execution Pipeline

### KNN Search Algorithm

```rust
impl KDTree {
    /// Find K nearest neighbors
    pub fn knn_search(
        &self,
        query: &[f32],
        vectors: &VectorStore,
        k: usize,
    ) -> Vec<(f32, usize)> {
        let mut heap = BoundedMaxHeap::new(k);
        
        if let Some(ref root) = self.root {
            self.search_recursive(
                root,
                query,
                vectors,
                &mut heap,
            );
        }
        
        heap.into_sorted_results()
    }
    
    fn search_recursive(
        &self,
        node: &KDNode,
        query: &[f32],
        vectors: &VectorStore,
        heap: &mut BoundedMaxHeap,
    ) {
        match node {
            KDNode::Leaf { indices } => {
                // Compute distances for all vectors in leaf
                for &idx in indices {
                    let vector = vectors.get(idx);
                    let distance = squared_euclidean_simd(query, vector);
                    heap.push(distance, idx);
                }
            }
            
            KDNode::Internal { 
                split_axis, 
                split_value, 
                left, 
                right 
            } => {
                // Determine which branch to explore first
                let query_value = query[*split_axis];
                let (first_branch, second_branch) = if query_value < *split_value {
                    (left, right)
                } else {
                    (right, left)
                };
                
                // Always explore the near branch
                self.search_recursive(first_branch, query, vectors, heap);
                
                // Check if far branch could contain closer points
                let axis_distance = (query_value - split_value).powi(2);
                
                if let Some(worst_dist) = heap.worst_distance() {
                    // Only search far branch if it intersects search sphere
                    if axis_distance < worst_dist || !heap.is_full() {
                        self.search_recursive(second_branch, query, vectors, heap);
                    }
                } else {
                    // Heap not full yet: must search far branch
                    self.search_recursive(second_branch, query, vectors, heap);
                }
            }
        }
    }
}
```

**Pruning Logic Visualization**:
```
Query Point: Q
Split Plane: | (vertical line at x=split_value)
Search Radius: r (defined by worst distance in heap)

Case 1: Far branch intersects search sphere
    ●────r────Q
         |
    Far  |  Near
   Branch|  Branch
         |
   Explore both branches

Case 2: Far branch outside search sphere
              ●─r─Q
         |
    Far  |  Near
   Branch|  Branch
         |
   Prune far branch (no closer points possible)
```

## SIMD Distance Computation

### Squared Euclidean Distance (L2)

```rust
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

/// Compute squared L2 distance using AVX2 intrinsics
#[cfg(target_feature = "avx2")]
pub unsafe fn squared_euclidean_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let mut sum = _mm256_setzero_ps();  // Accumulator: 8 f32 values
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        
        // Load 8 floats from each vector
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // Compute difference: (a - b)
        let diff = _mm256_sub_ps(va, vb);
        
        // Square the difference: (a - b)²
        // Use FMA: sum += diff * diff + 0
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal reduction: sum all 8 lanes
    let sum_reduced = horizontal_sum_avx2(sum);
    
    // Handle remaining elements (if len not multiple of 8)
    let remainder_start = chunks * 8;
    let remainder_sum: f32 = a[remainder_start..]
        .iter()
        .zip(&b[remainder_start..])
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    
    sum_reduced + remainder_sum
}

/// Horizontal sum of 8-lane AVX2 register
#[cfg(target_feature = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Split into two 128-bit halves
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    
    // Add halves together
    let sum128 = _mm_add_ps(low, high);
    
    // Further horizontal addition
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x01));
    
    // Extract scalar result
    _mm_cvtss_f32(sum32)
}
```

### Cosine Similarity

```rust
/// Compute cosine similarity: (a · b) / (||a|| * ||b||)
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = magnitude_simd(a);
    let norm_b = magnitude_simd(b);
    
    dot / (norm_a * norm_b)
}

/// Vector magnitude: sqrt(sum(x²))
#[cfg(target_feature = "avx2")]
pub unsafe fn magnitude_simd(v: &[f32]) -> f32 {
    let squared_sum = dot_product_simd(v, v);
    squared_sum.sqrt()
}
```

## Memory Layout Optimization

### Cache Line Awareness

**CPU Cache Hierarchy**:
```
L1 Cache: 32 KB per core, ~1 ns access
L2 Cache: 256 KB per core, ~3 ns access
L3 Cache: Shared, 8-32 MB, ~12 ns access
RAM: GBs, ~100 ns access
```

**Optimal Vector Alignment**:
```rust
/// Ensure vectors start on cache line boundaries
#[repr(align(64))]
pub struct AlignedVectorStore {
    vectors: Vec<f32>,
    dimension: usize,
}
```

**Prefetching**:
```rust
/// Hint to CPU to prefetch next cache line
#[cfg(target_arch = "x86_64")]
unsafe fn prefetch_next_vector(vectors: &[f32], index: usize, dimension: usize) {
    use std::arch::x86_64::*;
    let next_offset = (index + 1) * dimension;
    if next_offset < vectors.len() {
        let ptr = vectors.as_ptr().add(next_offset);
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}
```

## Error Handling & Recovery

### Graceful Degradation

```rust
pub enum SearchError {
    EmptyDatabase,
    InvalidDimension { expected: usize, got: usize },
    IndexNotBuilt,
    EmbeddingCallFailed(String),
}

impl VectorDatabase {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SearchError> {
        // Validation
        if self.vector_store.count() == 0 {
            return Err(SearchError::EmptyDatabase);
        }
        
        if query.len() != self.dimension {
            return Err(SearchError::InvalidDimension {
                expected: self.dimension,
                got: query.len(),
            });
        }
        
        // Fallback to brute-force if index not built
        if self.kd_tree.root.is_none() {
            return Ok(self.brute_force_search(query, k));
        }
        
        // Use index
        Ok(self.kd_tree.knn_search(query, &self.vector_store, k))
    }
}
```

## Concurrency Model

### Thread-Safe Operations

```rust
use std::sync::{Arc, RwLock};

pub struct ConcurrentVectorDatabase {
    /// Read-optimized: many readers, rare writers
    store: Arc<RwLock<VectorDatabase>>,
}

impl ConcurrentVectorDatabase {
    /// Multiple threads can search simultaneously
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SearchError> {
        let db = self.store.read().unwrap();
        db.search(query, k)
    }
    
    /// Exclusive write access for insertions
    pub fn insert(&self, vector: Vec<f32>, metadata: String) -> Result<u64, InsertError> {
        let mut db = self.store.write().unwrap();
        db.insert(vector, metadata)
    }
    
    /// Rebuild index in background thread
    pub fn rebuild_index_async(&self) {
        let store_clone = Arc::clone(&self.store);
        
        std::thread::spawn(move || {
            let mut db = store_clone.write().unwrap();
            db.rebuild_index();
        });
    }
}
```

## Performance Characteristics

### Time Complexity

| Operation | Brute Force | KD-Tree (Low-D) | KD-Tree (High-D) |
|-----------|-------------|-----------------|------------------|
| **Insert** | O(1) | O(log n) amortized | O(log n) |
| **Search** | O(n) | O(log n + k) | O(n) worst case |
| **Build Index** | N/A | O(n log n) | O(n log n) |
| **Memory** | O(n · d) | O(n · d + n) | O(n · d + n) |

### Space Complexity

```
Total Memory = VectorStore + RecordStore + KDTree + Heap

VectorStore:   n vectors × d dimensions × 4 bytes
RecordStore:   n records × ~100 bytes (with metadata)
KDTree:        n nodes × ~50 bytes
Search Heap:   k results × 12 bytes

Example (1M vectors, 768-dim):
VectorStore:   1M × 768 × 4 = 3.072 GB
RecordStore:   1M × 100 = 100 MB
KDTree:        1M × 50 = 50 MB
Total:         ~3.2 GB
```

---



# NanoVec: Cache Optimization & Memory Management

## CPU Cache Hierarchy

### Modern Cache Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CPU Core                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  L1 Data Cache: 32 KB                                 │ │
│  │  - Latency: ~1 ns (~4 cycles)                         │ │
│  │  - Bandwidth: ~400 GB/s                               │ │
│  │  - Cache Line: 64 bytes                               │ │
│  │  - Associativity: 8-way set associative               │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↕                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  L2 Cache: 256 KB (per core)                          │ │
│  │  - Latency: ~3 ns (~12 cycles)                        │ │
│  │  - Bandwidth: ~200 GB/s                               │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│  L3 Cache: 8-32 MB (shared across cores)                   │
│  - Latency: ~12 ns (~40 cycles)                             │
│  - Bandwidth: ~100 GB/s                                     │
└─────────────────────────┬───────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│  Main Memory (RAM): 16-64 GB                                │
│  - Latency: ~100 ns (~300 cycles)                           │
│  - Bandwidth: ~50 GB/s                                      │
└─────────────────────────────────────────────────────────────┘
```

### The Memory Wall

**Performance Gap**:
```
1980:  CPU/Memory speed ratio = 1:1
2000:  CPU/Memory speed ratio = 100:1
2025:  CPU/Memory speed ratio = 1000:1
```

**Implication**: CPU spends 90%+ of time waiting for data!

**Solution**: Exploit cache hierarchy through deliberate data layout.

---

## Cache Line Fundamentals

### Physical Reality

**Cache Line Size**: 64 bytes (universal across x86_64 and ARM)

**What Fits in One Cache Line**:
```rust
// Cache line = 64 bytes = 512 bits

// Scenario 1: Floating-point vectors
const FLOATS_PER_LINE: usize = 64 / 4;  // 16 f32 values

// Scenario 2: Mixed data (inefficient)
struct Document {
    id: u64,           // 8 bytes
    vector: Vec<f32>,  // 24 bytes (ptr + cap + len)
    metadata: String,  // 24 bytes (ptr + cap + len)
    // Total: 56 bytes → fits in one cache line
}
// BUT: vector/string data stored elsewhere on heap!
// Cache line contains POINTERS, not actual data → cache miss
```

**Memory Access Patterns**:
```
Scenario A: Sequential Access (GOOD)
Memory: [0][1][2][3][4][5][6][7][8][9]...
Access:  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
Result: Load [0-15], then [16-31], etc.
Cache hits: ~94% (only first access in each line misses)

Scenario B: Random Access (BAD)
Memory: [0][1][2][3][4][5][6][7][8][9]...
Access:  ↑     ↑        ↑  ↑     ↑     ↑
Result: Load [0-15], discard, load [48-63], discard...
Cache hits: ~0% (every access likely in different cache line)
```

---

## Data-Oriented Design Principles

### Anti-Pattern: Array of Structures (AoS)

```rust
/// Traditional OOP approach (CACHE INEFFICIENT)
struct Document {
    id: u64,                    // 8 bytes
    embedding: Vec<f32>,        // 24 bytes (heap pointer)
    metadata: String,           // 24 bytes (heap pointer)
    timestamp: u64,             // 8 bytes
    // Total: 64 bytes per Document
}

struct Database {
    documents: Vec<Document>,
}

// Memory layout on heap:
// [Doc0: id|ptr|ptr|ts][Doc1: id|ptr|ptr|ts][Doc2: id|ptr|ptr|ts]...
//         ↓             ↓
//     [768 floats]  [768 floats]  (scattered elsewhere in heap)
```

**Problem During Vector Scan**:
```rust
for doc in database.documents.iter() {
    let distance = compute_distance(query, &doc.embedding);
    // Cache line loads: id, ptr, ptr, timestamp (all USELESS for math)
    // Actual vector data: DIFFERENT cache line → MISS
    // Result: ~50% cache miss rate
}
```

**Cache Pollution**:
```
Cache Line 1: [Doc0.id | Doc0.vec_ptr | Doc0.meta_ptr | Doc0.timestamp]
              ↑ Used    ↑ Used          ↑ WASTED       ↑ WASTED

Efficiency: 16/64 bytes = 25% useful data
Wasted bandwidth: 75% of memory traffic is irrelevant metadata
```

---

### Optimal Pattern: Structure of Arrays (SoA)

```rust
/// Cache-optimal design (FAST)
pub struct VectorDatabase {
    /// Dense, contiguous f32 array: [v0_d0, v0_d1, ..., v0_dN, v1_d0, ...]
    vectors: Vec<f32>,
    
    /// Separate metadata storage (touched ONLY when returning results)
    records: Vec<VectorRecord>,
    
    dimension: usize,
}

struct VectorRecord {
    id: u64,
    vector_index: usize,  // Index into `vectors` array (NOT byte offset)
    metadata: String,
    timestamp: u64,
}
```

**Memory Layout**:
```
vectors array (contiguous):
  Cache Line 0: [v0_d0  v0_d1  v0_d2  ...  v0_d15 ]  ← 16 f32
  Cache Line 1: [v0_d16 v0_d17 v0_d18 ... v0_d31 ]
  ...
  Cache Line N: [v1_d0  v1_d1  v1_d2  ...  v1_d15 ]

records array (separate, rarely accessed):
  [Record0][Record1][Record2]...
```

**Performance During Vector Scan**:
```rust
// Iterate over raw f32 array (pure data, zero metadata)
for chunk in database.vectors.chunks_exact(dimension) {
    let distance = compute_distance_simd(query, chunk);
    // Cache line contains ONLY f32 data
    // Prefetcher predicts sequential access → 99% hit rate
}

// Only AFTER finding top-K results, retrieve metadata
for (distance, vector_idx) in top_k_results {
    let record = &database.records[vector_idx];
    println!("{}: {}", record.id, record.metadata);
}
```

**Cache Efficiency**:
```
Cache Line: [16 consecutive f32 values]
Efficiency: 64/64 bytes = 100% useful data
Bandwidth savings: 4x compared to AoS
```

---

## Memory Alignment

### Why Alignment Matters

**Aligned Access** (address % 64 == 0):
```
Memory: [........|XXXXXXXX|........]
         Cache 0  Cache 1  Cache 2
Load from aligned boundary → single cache line fetch
Cycles: 4-12 (L1 hit)
```

**Unaligned Access** (spans cache line boundary):
```
Memory: [....XXXX|XXXX....|........]
         Cache 0  Cache 1  Cache 2
Load spans two cache lines → TWO fetches required
Cycles: 8-24 (both lines must be fetched)
```

### Forcing Alignment in Rust

```rust
/// Align struct to 64-byte boundary (cache line size)
#[repr(align(64))]
pub struct AlignedVectorStore {
    vectors: Vec<f32>,
    dimension: usize,
}

// Verify alignment at runtime
fn check_alignment() {
    let store = AlignedVectorStore::new(768);
    let ptr = store.vectors.as_ptr() as usize;
    
    assert_eq!(ptr % 64, 0, "Vector storage not cache-aligned!");
}
```

**Manual Alignment with Allocator**:
```rust
use std::alloc::{alloc, Layout};

pub struct AlignedBuffer {
    ptr: *mut f32,
    len: usize,
    capacity: usize,
}

impl AlignedBuffer {
    pub fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<f32>(),
            64,  // Align to cache line
        ).unwrap();
        
        let ptr = unsafe { alloc(layout) as *mut f32 };
        
        Self { ptr, len: 0, capacity }
    }
}
```

---

## Cache Prefetching

### Hardware Prefetcher

**Automatic Prefetching**: Modern CPUs detect sequential access patterns.

```rust
// Sequential iteration → prefetcher activates
for i in 0..vectors.len() {
    process(vectors[i]);  // CPU prefetches vectors[i+1], [i+2], etc.
}

// Random access → prefetcher fails
for &idx in random_indices {
    process(vectors[idx]);  // Every access is a cache miss
}
```

**Stride Detection**:
```rust
// Prefetcher detects stride=768 (dimension)
for i in 0..num_vectors {
    let offset = i * 768;
    let vector = &vectors[offset..offset + 768];
    process(vector);
    // CPU prefetches vectors[offset + 768..offset + 1536]
}
```

### Manual Prefetching

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Explicitly hint CPU to prefetch next vector
pub unsafe fn prefetch_next_vector(
    vectors: &[f32],
    current_idx: usize,
    dimension: usize,
) {
    let next_offset = (current_idx + 1) * dimension;
    
    if next_offset < vectors.len() {
        let ptr = vectors.as_ptr().add(next_offset);
        
        // Prefetch to L1 cache (_MM_HINT_T0)
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

// Usage in search loop
for i in 0..num_vectors {
    unsafe { prefetch_next_vector(&db.vectors, i, dimension); }
    
    let vector = db.get_vector(i);
    let distance = compute_distance(query, vector);
}
```

**Prefetch Hints**:
```rust
_MM_HINT_T0  // Prefetch to L1 (temporal locality - will use soon)
_MM_HINT_T1  // Prefetch to L2 (moderate locality)
_MM_HINT_T2  // Prefetch to L3 (low locality)
_MM_HINT_NTA // Non-temporal (won't reuse - bypass cache)
```

---

## Cache-Aware Algorithms

### Blocking / Tiling

**Problem**: Matrix operations exceed cache size.

**Solution**: Process data in cache-sized blocks.

```rust
/// Matrix multiplication with cache blocking
pub fn matmul_blocked(
    a: &[f32],  // M × K
    b: &[f32],  // K × N
    c: &mut [f32],  // M × N
    m: usize,
    n: usize,
    k: usize,
) {
    const BLOCK_SIZE: usize = 64;  // Tune to L1 cache size
    
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for j_block in (0..n).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                // Process BLOCK_SIZE × BLOCK_SIZE sub-matrix
                for i in i_block..min(i_block + BLOCK_SIZE, m) {
                    for j in j_block..min(j_block + BLOCK_SIZE, n) {
                        let mut sum = c[i * n + j];
                        
                        for kk in k_block..min(k_block + BLOCK_SIZE, k) {
                            sum += a[i * k + kk] * b[kk * n + j];
                        }
                        
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}
```

**Effect**: Sub-matrices fit in L1, reducing memory traffic by 10-100x.

---

## False Sharing Prevention

### The Problem

**Scenario**: Multiple threads modifying adjacent memory.

```rust
// BAD: Counter array causes false sharing
struct Counters {
    counts: [AtomicU64; 8],  // 8 counters, 8 bytes each = 64 bytes total
}

// All 8 counters fit in ONE cache line!
// Thread 0 modifies counts[0] → invalidates entire cache line
// Thread 1 modifies counts[1] → must reload cache line
// Result: Cache line bounces between cores (coherency traffic)
```

**Cache Coherency Protocol** (MESI):
```
Core 0: Writes to counts[0]
  → Cache line marked "Modified"
  → Other cores' copies marked "Invalid"
  
Core 1: Reads counts[1] (same cache line!)
  → Must request line from Core 0
  → Core 0 flushes to memory
  → Core 1 loads line
  → Marked "Shared"

Result: Excessive coherency traffic, poor scaling
```

### Solution: Padding

```rust
/// Cache-line aligned counters (no false sharing)
#[repr(align(64))]
struct PaddedCounter {
    value: AtomicU64,
    _padding: [u8; 56],  // Force 64-byte alignment
}

struct Counters {
    counts: [PaddedCounter; 8],
}

// Now each counter occupies its own cache line
// Thread 0 modifies counts[0] → no impact on counts[1]'s cache line
// Threads can modify independently without coherency traffic
```

**Rust Helpers**:
```rust
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(align(64))]
pub struct CacheLinePadded<T> {
    value: T,
}

impl<T> CacheLinePadded<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
    
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

// Usage
let counters: Vec<CacheLinePadded<AtomicU64>> = 
    (0..8).map(|_| CacheLinePadded::new(AtomicU64::new(0))).collect();
```

---

## Memory Access Patterns

### Sequential vs. Random

**Benchmark**:
```rust
use std::time::Instant;

fn benchmark_access_patterns(size: usize) {
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    
    // Sequential access
    let start = Instant::now();
    let sum: f32 = data.iter().sum();
    let seq_time = start.elapsed();
    
    // Random access
    let mut rng = rand::thread_rng();
    let indices: Vec<usize> = (0..size).map(|_| rng.gen_range(0..size)).collect();
    
    let start = Instant::now();
    let sum: f32 = indices.iter().map(|&i| data[i]).sum();
    let rand_time = start.elapsed();
    
    println!("Sequential: {:?}", seq_time);
    println!("Random: {:?}", rand_time);
    println!("Slowdown: {:.2}x", rand_time.as_secs_f64() / seq_time.as_secs_f64());
}

// Typical results (size = 10M):
// Sequential: 8 ms
// Random: 150 ms
// Slowdown: 18.75x
```

### Improving Random Access

**Strategy 1: Sorting Indices**:
```rust
/// Sort indices before access to improve spatial locality
pub fn sorted_random_access(data: &[f32], mut indices: Vec<usize>) -> f32 {
    indices.sort_unstable();  // O(n log n) but improves cache hits
    
    indices.iter().map(|&i| data[i]).sum()
}
```

**Strategy 2: Batching**:
```rust
/// Process indices in batches that fit in cache
pub fn batched_access(data: &[f32], indices: &[usize]) -> f32 {
    const BATCH_SIZE: usize = 1024;  // Tune to L2 cache
    
    indices.chunks(BATCH_SIZE)
        .map(|batch| {
            batch.iter().map(|&i| data[i]).sum::<f32>()
        })
        .sum()
}
```

---

## SIMD and Cache Interaction

### Cache-Aligned SIMD Loads

```rust
#[cfg(target_feature = "avx2")]
unsafe fn aligned_vs_unaligned_load(data: &[f32]) {
    let ptr = data.as_ptr();
    
    // Aligned load (faster, requires 32-byte alignment)
    if ptr as usize % 32 == 0 {
        let vec = _mm256_load_ps(ptr);  // Aligned load instruction
    } else {
        let vec = _mm256_loadu_ps(ptr);  // Unaligned load (slower)
    }
}
```

**Performance**:
```
Aligned load (_mm256_load_ps):   1 cycle latency
Unaligned load (_mm256_loadu_ps): 1-3 cycles latency (if crosses cache line: +penalty)
```

### Non-Temporal Stores

**Use Case**: Writing large data that won't be reused.

```rust
#[cfg(target_feature = "avx2")]
unsafe fn stream_store(dest: &mut [f32], src: &[f32]) {
    assert_eq!(dest.len(), src.len());
    assert_eq!(dest.as_ptr() as usize % 32, 0, "Must be aligned");
    
    for i in (0..src.len()).step_by(8) {
        let data = _mm256_loadu_ps(src.as_ptr().add(i));
        
        // Non-temporal store: bypass cache, write directly to memory
        _mm256_stream_ps(dest.as_mut_ptr().add(i), data);
    }
    
    // Ensure all stores complete
    _mm_sfence();
}
```

**When to Use**:
- ✅ Large sequential writes (> L3 cache size)
- ✅ Data won't be read back immediately
- ❌ Small writes (cache bypass overhead outweighs benefit)
- ❌ Data will be accessed soon (forces memory fetch later)

---

## Profiling Cache Performance

### Using `perf` (Linux)

```bash
# Measure cache statistics
perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./nanovec

# Output:
#  1,234,567,890  cache-references
#     45,678,901  cache-misses              # 3.7% miss rate
#    123,456,789  L1-dcache-load-misses
```

**Interpreting Results**:
```
Good:  <5% L1 miss rate, <10% L3 miss rate
Fair:  5-10% L1, 10-20% L3
Poor:  >10% L1, >20% L3 → investigate data layout
```

### Using `valgrind --tool=cachegrind`

```bash
valgrind --tool=cachegrind --cache-sim=yes ./nanovec

# Generates cachegrind.out.<pid>
cg_annotate cachegrind.out.<pid>
```

**Output Analysis**:
```
Function: compute_distances
  I-cache:   99.9% hit rate  (instruction cache - usually good)
  D-cache:   87.3% hit rate  (data cache - could improve)
  LL-cache:  92.1% hit rate  (last-level cache)

→ Focus optimization on data cache misses
```

### Rust-Specific Tools

```rust
// Criterion benchmark with cache profiling
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_vector_scan(c: &mut Criterion) {
    let db = create_database(10_000, 768);
    let query = create_random_vector(768);
    
    c.bench_function("vector_scan", |b| {
        b.iter(|| {
            // black_box prevents compiler optimization
            let results = db.search(black_box(&query), black_box(10));
            black_box(results)
        })
    });
}

criterion_group!(benches, bench_vector_scan);
criterion_main!(benches);
```

---

## Best Practices Summary

### ✅ DO

1. **Use contiguous arrays** (`Vec<T>`) over linked structures
2. **Separate hot and cold data** (SoA over AoS)
3. **Iterate sequentially** whenever possible
4. **Align data** to cache line boundaries (64 bytes)
5. **Prefetch** when random access is unavoidable
6. **Profile first** before optimizing

### ❌ DON'T

1. **Mix metadata with math data** in same struct
2. **Use HashMap** for hot paths (unpredictable access pattern)
3. **Allocate small objects** repeatedly (heap fragmentation)
4. **Ignore alignment** requirements for SIMD
5. **Assume compiler optimizes** everything automatically
6. **Over-engineer** without measuring impact

---

## Case Study: NanoVec Vector Scan

### Before Optimization (AoS)

```rust
struct Document {
    id: u64,
    embedding: Vec<f32>,  // Heap allocation
    metadata: String,
}

// Cache miss per document
for doc in documents {
    compute_distance(query, &doc.embedding);  // Chases pointer to heap
}

// Performance: 50% cache miss rate, 12 ms for 10K vectors
```

### After Optimization (SoA)

```rust
struct VectorStore {
    vectors: Vec<f32>,  // Flat, contiguous
    dimension: usize,
}

// Sequential scan with perfect spatial locality
for chunk in vectors.chunks_exact(dimension) {
    compute_distance_simd(query, chunk);  // Pure f32 data
}

// Performance: 5% cache miss rate, 2 ms for 10K vectors
// Speedup: 6x from cache optimization alone!
```

---

**Next**: Integration with MCP and deployment in `06-MCP-INTEGRATION.md`

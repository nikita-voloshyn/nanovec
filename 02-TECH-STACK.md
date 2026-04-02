# NanoVec: Technology Stack & Dependencies

## Core Technology Stack

### Primary Language: Rust

**Version**: 1.75+ (stable channel for production, nightly for SIMD features)

**Why Rust?**
- **Memory Safety**: Ownership model prevents segmentation faults at compile-time
- **Zero-Cost Abstractions**: Performance equivalent to C/C++ without manual memory management
- **Fearless Concurrency**: Thread-safety guaranteed by the borrow checker
- **LLVM Backend**: Access to advanced optimization passes and SIMD intrinsics
- **Growing AI Ecosystem**: Emerging as the infrastructure language for AI systems

### Build System

**Cargo**: Rust's native package manager and build system
- Project configuration via `Cargo.toml`
- Dependency management
- Build profiles (dev, release, bench)
- Integrated testing framework

## Core Dependencies

### Zero External Indexing Libraries

**Critical Design Decision**: No FAISS, HNSWLIB, or other vector search libraries

**Rationale**:
- Deep understanding through first-principles implementation
- Full control over memory layouts and algorithms
- Educational value of building from scratch
- Elimination of dependency bloat

### Essential Crates (Minimal Set)

#### 1. Model Context Protocol Integration

```toml
[dependencies]
# Official Rust MCP SDK
rmcp = "0.1"  # Model Context Protocol server implementation
tokio = { version = "1.35", features = ["full"] }  # Async runtime for MCP
serde = { version = "1.0", features = ["derive"] }  # JSON serialization
serde_json = "1.0"  # JSON-RPC 2.0 message formatting
```

**Purpose**:
- `rmcp`: Native MCP server capabilities
- `tokio`: Async I/O for stdio/SSE transports
- `serde`: Schema serialization for tool definitions

#### 2. Mathematical Foundations

```toml
[dependencies]
# NO external vector libraries - implementing from scratch
# Using only std::simd (nightly feature) or std::arch intrinsics
```

**SIMD Options**:

**Option A - Portable SIMD (Nightly)**:
```toml
[dependencies]
# Requires nightly toolchain
# Enable in lib.rs: #![feature(portable_simd)]
```

**Option B - Platform-Specific Intrinsics**:
```rust
// No dependencies - using std::arch::x86_64 or std::arch::aarch64
// Conditional compilation based on target architecture
```

#### 3. Testing & Benchmarking

```toml
[dev-dependencies]
criterion = "0.5"  # Statistical benchmarking
proptest = "1.4"   # Property-based testing
divan = "0.1"      # Lightweight alternative to criterion
approx = "0.5"     # Floating-point comparison utilities
```

**Purpose**:
- Verify SIMD speedups vs. scalar implementations
- Property-based testing for KD-Tree correctness
- Microbenchmarks for memory layout strategies

#### 4. Development & Debugging

```toml
[dev-dependencies]
tracing = "0.1"           # Structured logging
tracing-subscriber = "0.3" # Log formatting
cargo-show-asm = "0.2"    # Inspect generated assembly
```

**Purpose**:
- Debug MCP stdio communication (route to stderr)
- Verify SIMD codegen with assembly inspection
- Performance profiling

## Optional Enhancement Crates

### For Production SSE Transport

```toml
[dependencies]
# Only if implementing Server-Sent Events transport
axum = "0.7"              # Web framework
tower-http = "0.5"        # HTTP middleware
oauth2 = "4.4"            # OAuth 2.1 authentication
```

### For Advanced Embeddings

```toml
[dependencies]
# If integrating local embedding models
candle-core = "0.3"       # Rust ML framework
tokenizers = "0.15"       # HuggingFace tokenizers
```

## Compiler Configuration

### Cargo.toml Build Profiles

```toml
[profile.dev]
opt-level = 0        # No optimization for fast compile times

[profile.release]
opt-level = 3        # Maximum optimization
lto = "fat"          # Link-Time Optimization
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Smaller binary, faster panic handling
strip = true         # Strip debug symbols

[profile.bench]
inherits = "release"
debug = true         # Keep debug symbols for profiler
```

### Target-Specific Compilation

```bash
# For x86_64 with AVX2 support
RUSTFLAGS="-C target-cpu=native" cargo build --release

# For Apple Silicon (M1/M2)
RUSTFLAGS="-C target-cpu=native" cargo build --release --target aarch64-apple-darwin

# For production (conservative compatibility)
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

## SIMD Implementation Strategies

### Strategy 1: Nightly Portable SIMD

**Activation**:
```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly"
```

```rust
// src/lib.rs
#![feature(portable_simd)]

use std::simd::{f32x8, SimdFloat};
```

**Pros**:
- ✅ Write once, runs on x86_64 (AVX2) and ARM (NEON)
- ✅ High-level API with safety guarantees
- ✅ Future-proof as feature stabilizes

**Cons**:
- ❌ Requires nightly Rust
- ❌ Slightly less control than intrinsics

### Strategy 2: Platform-Specific Intrinsics

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
```

**Pros**:
- ✅ Works on stable Rust
- ✅ Maximum performance control
- ✅ Direct mapping to CPU instructions

**Cons**:
- ❌ Separate implementations per architecture
- ❌ More complex codebase

### Strategy 3: Hybrid Approach (Recommended)

```rust
// Fallback scalar implementation
#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// AVX2 implementation for x86_64
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Use __m256 and _mm256_fmadd_ps
}

// NEON implementation for ARM
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Use float32x4_t and vmlaq_f32
}
```

## Platform Support Matrix

| Platform | Architecture | SIMD Support | Status |
|----------|-------------|--------------|---------|
| **Linux x86_64** | Intel/AMD | AVX2/FMA | ✅ Primary |
| **macOS ARM64** | Apple Silicon | NEON | ✅ Supported |
| **Windows x86_64** | Intel/AMD | AVX2/FMA | ✅ Supported |
| **Linux ARM64** | AWS Graviton | NEON | ✅ Supported |
| **WebAssembly** | WASM SIMD | 128-bit | 🔄 Experimental |

## Development Tools

### Required Tools

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install nightly for SIMD development
rustup toolchain install nightly
rustup default nightly

# Install assembly viewer
cargo install cargo-show-asm

# Install benchmarking tools
cargo install cargo-criterion
```

### Optional Tools

```bash
# Profiling
cargo install flamegraph
cargo install cargo-instruments  # macOS only

# Code quality
cargo install cargo-audit        # Security audits
cargo install cargo-outdated     # Dependency updates
cargo install cargo-deny         # Dependency linting
```

## Testing Infrastructure

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product_accuracy() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = dot_product(&a, &b);
        assert_relative_eq!(result, 70.0, epsilon = 1e-6);
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_simd_matches_scalar(
        a in prop::collection::vec(-1000.0f32..1000.0, 768),
        b in prop::collection::vec(-1000.0f32..1000.0, 768)
    ) {
        let scalar_result = dot_product_scalar(&a, &b);
        let simd_result = dot_product_simd(&a, &b);
        
        assert_relative_eq!(scalar_result, simd_result, epsilon = 1e-4);
    }
}
```

### Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_dot_product(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..768).map(|i| (i * 2) as f32).collect();
    
    c.bench_function("scalar_dot_product", |bencher| {
        bencher.iter(|| dot_product_scalar(black_box(&a), black_box(&b)))
    });
    
    c.bench_function("simd_dot_product", |bencher| {
        bencher.iter(|| dot_product_simd(black_box(&a), black_box(&b)))
    });
}

criterion_group!(benches, benchmark_dot_product);
criterion_main!(benches);
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, nightly]
    
    steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
    
    - name: Run tests
      run: cargo test --all-features
    
    - name: Run benchmarks
      run: cargo bench --no-run
```

## Security & Compliance

### Dependency Auditing

```bash
# Check for security vulnerabilities
cargo audit

# Verify license compliance
cargo deny check licenses
```

### Memory Safety Verification

```bash
# Run with address sanitizer
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test

# Run with memory sanitizer
RUSTFLAGS="-Z sanitizer=memory" cargo +nightly test
```

## Deployment Artifacts

### Binary Distribution

```bash
# Single static binary
cargo build --release --target x86_64-unknown-linux-musl
strip target/x86_64-unknown-linux-musl/release/nanovec

# macOS universal binary
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create -output nanovec \
  target/x86_64-apple-darwin/release/nanovec \
  target/aarch64-apple-darwin/release/nanovec
```

### Container Image

```dockerfile
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /build
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/nanovec /
ENTRYPOINT ["/nanovec"]
```

## Summary

NanoVec's technology stack prioritizes:
1. **Minimal Dependencies**: Zero indexing libraries, only essential infrastructure
2. **Maximum Performance**: SIMD operations via portable or intrinsic APIs
3. **Platform Agnostic**: Conditional compilation for x86_64 and ARM
4. **Developer Experience**: Comprehensive testing and benchmarking tools
5. **Production Ready**: Security auditing and static binary deployment

The stack is intentionally lean to maintain full control over performance characteristics while demonstrating mastery of low-level systems programming in Rust.

---



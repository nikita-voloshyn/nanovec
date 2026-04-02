---
name: bench
description: "Run criterion/divan benchmarks for NanoVec. Compare scalar vs SIMD performance, generate reports, and track regressions. Triggers: benchmark, perf, bench, simd vs scalar, performance."
---

# Bench: Performance Measurement

## Steps

1. **Identify target.** Determine what to benchmark:
   - `all` -- run all benchmarks
   - `distance` -- distance function benchmarks (scalar vs SIMD)
   - `search` -- search benchmarks (brute-force vs KD-Tree)
   - A specific function name

2. **Set CPU target.** Ensure SIMD instructions are enabled:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo bench 2>&1
   ```

3. **Run benchmarks.** Execute the appropriate benchmark suite:
   ```bash
   # All benchmarks
   RUSTFLAGS="-C target-cpu=native" cargo bench

   # Specific benchmark group
   RUSTFLAGS="-C target-cpu=native" cargo bench -- dot_product

   # With criterion HTML report
   RUSTFLAGS="-C target-cpu=native" cargo bench -- --output-format=bencher
   ```

4. **Compare results.** If comparing scalar vs SIMD:
   - Extract scalar baseline time (ns/iter)
   - Extract SIMD optimized time (ns/iter)
   - Calculate speedup factor: `scalar_time / simd_time`
   - Report: "AVX2 dot product: Xns vs Yns (Z.Zx speedup)"

5. **Check for regressions.** Compare against previous benchmark results:
   ```bash
   # Criterion stores baselines in target/criterion/
   ls target/criterion/ 2>/dev/null
   ```

6. **Generate report.** Format results as:
   ```markdown
   ## Benchmark Report: <date>

   ### Environment
   - CPU: <architecture>
   - SIMD: <AVX2/NEON/scalar>
   - Rust: <toolchain version>

   ### Results

   | Function | Scalar (ns) | SIMD (ns) | Speedup |
   |----------|-------------|-----------|---------|
   | dot_product (768-dim) | X | Y | Z.Zx |
   | euclidean_sq (768-dim) | X | Y | Z.Zx |
   | cosine_sim (768-dim) | X | Y | Z.Zx |

   ### Analysis
   <observations about performance characteristics>
   ```

---
name: asm
description: "Inspect generated assembly for NanoVec functions via cargo-show-asm. Use to verify SIMD instructions are emitted correctly, check for unexpected scalar fallbacks, or analyze codegen quality. Triggers: assembly, asm, codegen, show asm, instruction."
---

# ASM: Assembly Inspection

## Steps

1. **Identify target function.** Determine which function to inspect. Common targets:
   - `nanovec::simd::avx2::dot_product_avx2`
   - `nanovec::simd::neon::dot_product_neon`
   - `nanovec::simd::portable::dot_product_portable`
   - `nanovec::distance::euclidean::euclidean_squared`

2. **Run cargo-show-asm.** Inspect the generated assembly:
   ```bash
   # With native CPU features enabled
   RUSTFLAGS="-C target-cpu=native" cargo asm <function_path>

   # For a specific target
   RUSTFLAGS="-C target-cpu=x86-64-v3" cargo asm <function_path>

   # List available functions matching a pattern
   cargo asm --lib | grep dot_product
   ```

3. **Verify expected instructions.** Check for:

   **AVX2 functions should contain:**
   - `vmovups` or `vmovaps` -- 256-bit loads
   - `vfmadd231ps` or `vfmadd213ps` -- FMA operations
   - `vsubps` -- 256-bit subtraction
   - `vhaddps` or manual horizontal reduction

   **NEON functions should contain:**
   - `ldp` or `ldr` -- vector loads
   - `fmla` -- fused multiply-add
   - `fsub` -- vector subtraction
   - `faddp` -- pairwise addition

   **Red flags (unexpected):**
   - `call` instructions in inner loops (function call overhead)
   - Scalar `mulss`/`addss` where SIMD was expected
   - `movss` where `vmovups` was expected (scalar fallback)

4. **Compare codegen.** If comparing two implementations:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo asm nanovec::distance::euclidean_squared > /tmp/scalar.asm
   RUSTFLAGS="-C target-cpu=native" cargo asm nanovec::simd::avx2::euclidean_squared_avx2 > /tmp/simd.asm
   diff /tmp/scalar.asm /tmp/simd.asm
   ```

5. **Report findings.** Document:
   - Expected vs actual instructions
   - Instruction count in the hot loop
   - Any unexpected scalar fallbacks
   - Optimization opportunities (e.g., missing FMA, unnecessary register spills)

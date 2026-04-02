---
name: check
description: "Full quality pipeline for NanoVec: format check, clippy lint, and test suite. Use before commits, after implementation, or to verify project health. Triggers: check, quality, ci, lint, verify."
---

# Check: Quality Pipeline

## Steps

1. **Format check.** Verify all Rust files conform to rustfmt standards:
   ```bash
   cargo fmt --check
   ```
   If this fails, run `cargo fmt` to fix, then re-check.

2. **Clippy lint.** Run clippy with all warnings treated as errors:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```
   If this fails, fix each warning before proceeding. Common fixes:
   - `clippy::needless_return` -- remove explicit `return` at end of function
   - `clippy::redundant_clone` -- remove unnecessary `.clone()`
   - `clippy::manual_map` -- use `.map()` instead of `match`

3. **Test suite.** Run all tests including doc tests:
   ```bash
   cargo test --all-features
   ```
   If any test fails, report the failure with the full test output and do not proceed.

4. **Summary.** Report results:
   ```
   Format:  PASS/FAIL
   Clippy:  PASS/FAIL (N warnings fixed)
   Tests:   PASS/FAIL (N passed, M failed)
   ```

5. **Unsafe audit.** Check for unsafe blocks without SAFETY comments:
   ```bash
   grep -rn 'unsafe' src/ --include='*.rs' | grep -v '// SAFETY:' | grep -v '#\[cfg' | grep -v 'unsafe_code'
   ```
   Report any matches as warnings that need SAFETY documentation.

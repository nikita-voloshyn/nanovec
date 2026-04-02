---
name: audit
description: "Security and safety audit for NanoVec: cargo audit for vulnerabilities, cargo deny for license compliance, and sanitizer runs (AddressSanitizer, MemorySanitizer) for memory safety verification. Triggers: audit, security, sanitizer, memory safety, vulnerability."
---

# Audit: Security and Safety

## Steps

1. **Dependency vulnerability scan.**
   ```bash
   cargo audit 2>&1
   ```
   If vulnerabilities are found, report severity level and affected crate. Suggest upgrade path or alternative.

2. **License compliance check.**
   ```bash
   cargo deny check licenses 2>&1
   ```
   NanoVec targets MIT/Apache-2.0 dual license. Flag any dependency with incompatible licenses (GPL, AGPL, proprietary).

3. **AddressSanitizer run.** Detect buffer overflows, use-after-free, and memory leaks:
   ```bash
   RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --all-features --target x86_64-unknown-linux-gnu 2>&1
   ```
   On macOS:
   ```bash
   RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --all-features --target aarch64-apple-darwin 2>&1
   ```

4. **MemorySanitizer run.** Detect uninitialized memory reads:
   ```bash
   RUSTFLAGS="-Z sanitizer=memory" cargo +nightly test --all-features --target x86_64-unknown-linux-gnu 2>&1
   ```
   Note: MemorySanitizer requires Linux x86_64 with nightly Rust.

5. **Unsafe code audit.** Enumerate all unsafe blocks and verify documentation:
   ```bash
   grep -rn 'unsafe' src/ --include='*.rs'
   ```
   For each unsafe block, verify:
   - A `// SAFETY:` comment exists immediately above or inline
   - The safety invariant is specific (not "this is safe because I checked")
   - The invariant is actually maintained by surrounding code

6. **Report.**
   ```
   Vulnerabilities:     N found (X critical, Y moderate, Z low)
   License issues:      N found
   AddressSanitizer:    PASS/FAIL
   MemorySanitizer:     PASS/FAIL/SKIPPED (Linux only)
   Unsafe blocks:       N total, M without SAFETY docs

   Recommendations:
   - <actionable items>
   ```

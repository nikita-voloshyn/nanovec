---
name: deploy-check
description: "Pre-deployment audit checklist for NanoVec releases. Verifies build, tests, security, documentation, and binary artifacts. Triggers: deploy check, release ready, pre-release, ship it."
---

# Deploy Check: Pre-Release Audit

## Steps

1. **Build check.** Verify release build compiles cleanly:
   ```bash
   cargo build --release 2>&1 | tail -10
   ```

2. **Quality pipeline.** Run the full quality gate:
   ```bash
   cargo fmt --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   ```

3. **Security audit.** Check for known vulnerabilities:
   ```bash
   cargo audit 2>&1 || echo "cargo-audit not installed, skipping"
   ```

4. **License check.** Verify dependency licenses:
   ```bash
   cargo deny check licenses 2>&1 || echo "cargo-deny not installed, skipping"
   ```

5. **Unsafe audit.** List all unsafe blocks and verify SAFETY comments:
   ```bash
   grep -rn 'unsafe' src/ --include='*.rs' | grep -v '// SAFETY:' | grep -v '#\[cfg' | grep -v 'unsafe_code'
   ```

6. **Binary size.** Check release binary size:
   ```bash
   ls -lh target/release/nanovec 2>/dev/null || echo "Binary not found"
   ```

7. **Documentation check.** Verify docs are up to date:
   ```bash
   ls docs/components/ 2>/dev/null
   cat docs/coverage.md 2>/dev/null || echo "No coverage report"
   ```

8. **CHANGELOG check.** Verify changelog has an entry for this release:
   ```bash
   head -20 CHANGELOG.md 2>/dev/null || echo "No CHANGELOG.md"
   ```

9. **Summary.** Report audit results:
   ```
   Build:          PASS/FAIL
   Quality:        PASS/FAIL
   Security:       PASS/FAIL/SKIPPED
   Licenses:       PASS/FAIL/SKIPPED
   Unsafe blocks:  N total, M without SAFETY
   Binary size:    X MB
   Docs coverage:  X%
   CHANGELOG:      up to date / needs update
   ```

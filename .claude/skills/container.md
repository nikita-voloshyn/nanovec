---
name: container
description: "Docker multi-arch build for NanoVec: linux-musl static binary and macOS universal binary. Use to prepare release artifacts or test containerized deployment. Triggers: docker, container, deploy, build image, release binary."
---

# Container: Multi-Arch Build

## Steps

1. **Check prerequisites.**
   ```bash
   rustup target list --installed
   docker --version 2>/dev/null || echo "Docker not installed"
   ```

2. **Build static Linux binary (musl).**
   ```bash
   # Install musl target if needed
   rustup target add x86_64-unknown-linux-musl

   # Build static binary
   RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-unknown-linux-musl

   # Verify it is statically linked
   file target/x86_64-unknown-linux-musl/release/nanovec
   ```

3. **Build macOS universal binary (if on macOS).**
   ```bash
   # Build for both architectures
   cargo build --release --target x86_64-apple-darwin
   cargo build --release --target aarch64-apple-darwin

   # Create universal binary
   lipo -create \
     target/x86_64-apple-darwin/release/nanovec \
     target/aarch64-apple-darwin/release/nanovec \
     -output target/nanovec-universal
   ```

4. **Build Docker image.**
   ```dockerfile
   # Dockerfile
   FROM rust:1.75-alpine AS builder
   RUN apk add --no-cache musl-dev
   WORKDIR /build
   COPY . .
   RUN RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-unknown-linux-musl

   FROM scratch
   COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/nanovec /nanovec
   ENTRYPOINT ["/nanovec"]
   ```

   ```bash
   docker build -t nanovec:latest .
   docker images nanovec
   ```

5. **Verify container.**
   ```bash
   # Test the container starts and responds to MCP stdio
   echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | docker run -i nanovec:latest
   ```

6. **Report artifacts.**
   ```
   Linux (musl):  target/x86_64-unknown-linux-musl/release/nanovec  (X MB)
   macOS (universal): target/nanovec-universal  (X MB)
   Docker image:  nanovec:latest  (X MB)
   ```

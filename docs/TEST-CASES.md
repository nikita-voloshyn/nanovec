# NanoVec Phase 1 — Test Cases

## Setup

NanoVec is a pure vector store. It does **not** embed text — you supply pre-computed float
vectors. The test cases below use hand-crafted 3-dimensional unit vectors to keep things
readable while still exercising real geometric similarity logic.

**Vector semantics used in these tests:**

```
Axis 0 (x): systems / low-level programming
Axis 1 (y): data science / machine learning
Axis 2 (z): web / frontend development
```

Examples:
- `[1.0, 0.0, 0.0]` → pure systems topic
- `[0.0, 1.0, 0.0]` → pure data science topic
- `[0.7, 0.7, 0.0]` → systems + data science (normalized ≈ [0.71, 0.71, 0])
- `[0.0, 0.0, 1.0]` → pure web topic

---

## TC-01: Basic Index and Search

**Goal:** Verify that `index_vector` stores a document and `search` finds it.

**Steps:**

1. Index one document:
   ```
   index_vector(
     text     = "Rust systems programming language",
     vector   = [1.0, 0.0, 0.0]
   )
   ```
   **Expected:** `{ "id": 0 }`

2. Search with the same vector:
   ```
   search(
     vector = [1.0, 0.0, 0.0],
     k      = 1
   )
   ```
   **Expected:**
   ```json
   [{ "id": 0, "score": 0.0, "text": "Rust systems programming language" }]
   ```
   Score = 0.0 means identical direction (Euclidean distance from a vector to itself is 0).

---

## TC-02: Nearest Neighbour Ranking

**Goal:** Verify that `search` returns results in correct similarity order.

**Steps:**

1. Index three documents:
   ```
   index_vector("Rust systems programming",   [1.0, 0.0, 0.0])  → id 0
   index_vector("C++ embedded systems",       [0.9, 0.1, 0.0])  → id 1
   index_vector("React web components",       [0.0, 0.0, 1.0])  → id 2
   ```

2. Search for the most systems-like topic:
   ```
   search(vector=[1.0, 0.0, 0.0], k=3)
   ```

**Expected order (ascending score = closest first):**

| Rank | id | text                       | score (approx) |
|------|----|----------------------------|----------------|
| 1    | 0  | Rust systems programming   | 0.000          |
| 2    | 1  | C++ embedded systems       | ~0.100         |
| 3    | 2  | React web components       | ~1.414         |

React scores ~1.414 because it is orthogonal to the query vector (maximum Euclidean distance
for unit vectors is √2 ≈ 1.414).

---

## TC-03: Delete and Verify Removal

**Goal:** Verify that `delete` removes a document and it no longer appears in search results.

**Steps:**

1. Index two documents:
   ```
   index_vector("Rust systems programming", [1.0, 0.0, 0.0])  → id 0
   index_vector("Go backend services",      [0.8, 0.0, 0.2])  → id 1
   ```

2. Delete the first document:
   ```
   delete(id=0)
   ```
   **Expected:** `{ "success": true }`

3. Search with the original query:
   ```
   search(vector=[1.0, 0.0, 0.0], k=5)
   ```
   **Expected:** Only id 1 appears. id 0 is gone.

4. Verify stats reflect the deletion:
   ```
   stats()
   ```
   **Expected:** `{ "count": 1, "dimension": 3, "metric": "Euclidean" }`

---

## TC-04: Stats — Empty Store

**Goal:** Verify `stats` reports correct state before any documents are indexed.

**Steps:**

1. Call stats on a fresh server:
   ```
   stats()
   ```
   **Expected:**
   ```json
   { "count": 0, "dimension": null, "metric": "Euclidean" }
   ```
   `dimension` is `null` because no vector has been indexed yet — dimension is set on first
   `index_vector` call.

---

## TC-05: Dimension Lock

**Goal:** Verify that all vectors must have the same dimension after the first insert.

**Steps:**

1. Index a 3-dimensional vector (sets dimension = 3):
   ```
   index_vector("First document", [1.0, 0.0, 0.0])
   ```
   **Expected:** `{ "id": 0 }`

2. Attempt to index a 4-dimensional vector:
   ```
   index_vector("Second document", [1.0, 0.0, 0.0, 0.5])
   ```
   **Expected:** Error — `"dimension mismatch: expected 3, got 4"`

---

## TC-06: Metric Variants

**Goal:** Verify that all three distance metrics work and produce valid results.

**Setup:** Index two documents:
```
index_vector("Topic A", [1.0, 0.0, 0.0])
index_vector("Topic B", [0.0, 1.0, 0.0])
```

**Euclidean search:**
```
search(vector=[1.0, 0.0, 0.0], k=2, metric="euclidean")
```
Expected: Topic A first (score ≈ 0.0), Topic B second (score ≈ 1.414)

**Cosine search:**
```
search(vector=[1.0, 0.0, 0.0], k=2, metric="cosine")
```
Expected: Topic A first (score ≈ 0.0), Topic B second (score ≈ 1.0)
Cosine distance between orthogonal unit vectors = 1 - cos(90°) = 1.0.

**Dot product search:**
```
search(vector=[1.0, 0.0, 0.0], k=2, metric="dot")
```
Expected: Topic A first (score ≈ -1.0 as stored negated for min-heap), Topic B second.
Note: dot product is a similarity (higher = better), so NanoVec internally negates it to
fit the "lower score = closer" convention used by the heap.

---

## TC-07: Full Agent Memory Workflow

**Goal:** Simulate a realistic use case — an AI agent storing and retrieving working memory.

**Scenario:** An agent is working on a Rust project and accumulates facts across a session.

**Steps:**

1. Store session facts:
   ```
   index_vector("We are using rmcp 1.3.0 for MCP transport",   [0.9, 0.0, 0.1])
   index_vector("Cargo.toml edition is 2021",                  [0.8, 0.0, 0.0])
   index_vector("Tests use proptest for property-based checks", [0.7, 0.3, 0.0])
   index_vector("Target is sub-millisecond search latency",     [0.6, 0.4, 0.0])
   index_vector("React frontend uses shadcn/ui components",     [0.0, 0.0, 1.0])
   ```

2. Retrieve context relevant to "Rust project technical details":
   ```
   search(vector=[0.85, 0.1, 0.0], k=3)
   ```

   **Expected top 3 (all Rust/systems facts, React excluded):**
   - "Cargo.toml edition is 2021"
   - "We are using rmcp 1.3.0 for MCP transport"
   - "Tests use proptest for property-based checks"

   The React entry should not appear in top 3 — its vector `[0.0, 0.0, 1.0]` is far from
   the query.

3. Check memory usage:
   ```
   stats()
   ```
   **Expected:** `{ "count": 5, "dimension": 3, "metric": "Euclidean" }`

---

## Running via Cargo Integration Test

All of TC-01 through TC-03 are covered by the automated integration test:

```bash
cargo test test_mcp_full_flow --test integration -- --nocapture
```

Expected output:
```
running 1 test
test mcp_stdio::test_mcp_full_flow ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

## Running Live via Claude Code MCP

After restarting Claude Code, ask:

> "Use the nanovec MCP tool to run test case TC-02: index three documents about Rust,
> C++, and React with vectors [1,0,0], [0.9,0.1,0], [0,0,1], then search with k=3 and
> show me the ranking."

NanoVec will appear in the available tools panel once Claude Code reloads `~/.claude/mcp.json`.

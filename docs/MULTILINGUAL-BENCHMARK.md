# Multilingual benchmark — `all-MiniLM-L6-v2`

**Date:** 2026-05-07
**Corpus:** [`TEST-CORPUS-MULTILINGUAL.md`](./TEST-CORPUS-MULTILINGUAL.md) — 10 ScootGo FAQ topics × 3 languages (en/pl/uk) = 30 chunks
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, primarily English-trained)
**Metric:** cosine (server default for `search_document`)
**Procedure:** for each topic, issue one parallel query in each language with `k=5`, record whether the matching topic appears in the top-5 and at what rank.

## Topic-language ID map

| Topic # | Topic | en | pl | uk |
|--------:|-------|---:|---:|---:|
| 1 | unlocking | 0 | 1 | 2 |
| 2 | payment | 3 | 4 | 5 |
| 3 | speed_limits | 6 | 7 | 8 |
| 4 | helmet | 9 | 10 | 11 |
| 5 | parking | 12 | 13 | 14 |
| 6 | battery | 15 | 16 | 17 |
| 7 | damage | 18 | 19 | 20 |
| 8 | support_hours | 21 | 22 | 23 |
| 9 | refunds | 24 | 25 | 26 |
| 10 | safety | 27 | 28 | 29 |

## Results — recall@5 per language

Five test topics queried in each language. Cell shows rank and similarity of the correct same-language chunk; `❌` means absent from top-5.

| Topic | PL | EN | UK |
|-------|----|----|-----|
| payment | ❌ | ❌ | ❌ |
| helmet | #3 (0.49) | **#1 (0.60)** | #3 (0.58) |
| parking | #4 (0.63) | #4 (0.43) | ❌ |
| battery | #4 (0.62) | **#1 (0.50)** + cross-lingual PL @ #4 | #2 (0.61) |
| damage | #4 (0.59) | #4 (0.39) | #3 (0.59) |
| **Recall@5** | **4/5 (80%)** | **4/5 (80%)** | **3/5 (60%)** |
| **#1 hits** | 0 | 2 | 0 |
| **Cross-lingual hits in top-5** | 0 | 1 | 0 |

## Findings

### 1. Cross-lingual transfer is effectively zero

Across 15 queries (5 topics × 3 languages), exactly **one** cross-lingual hit was observed: the EN query *"How many kilometers does the battery last?"* surfaced `id=16` (battery-pl) at rank #4. The trigger was the shared numeric/Latin token `30 km` / `25 km/h`, which appears identically in both language chunks.

The model assigns each language to a roughly disjoint region of embedding space. Without shared surface tokens (numbers, brand names, Latin loanwords), embeddings of semantically equivalent EN/PL/UK sentences do not retrieve each other.

### 2. EN has stronger ranking discrimination

EN is the only language where the correct topic landed at **rank #1** (helmet, battery). PL and UK never produced a #1 hit despite higher absolute cosine similarities. This indicates that EN embeddings are spread out enough for the correct chunk to dominate, while PL/UK embeddings cluster more tightly around dominant chunks.

### 3. Per-language attractor chunks distort ranking

- **PL:** `id=1` (unlocking-pl) is a strong attractor — appears in top-5 for 4/5 PL queries, including unrelated ones (battery, damage, parking).
- **UK:** `id=29` (safety-uk) is an even stronger attractor — appears at **rank #1 for all 5 UK queries**, regardless of topic.
- **EN:** no comparable global attractor; results are more topic-aligned.

This pattern is consistent with the model having less training data per non-English language: the resulting embeddings cluster more tightly, and chunks with broader vocabulary "absorb" any query in that language.

### 4. The `payment` topic fails in all three languages — corpus issue, not language issue

No language retrieves the `payment` chunk in top-5 for the natural query *"how do I pay?"*. The chunk text is a list of card brands (Visa, Mastercard, Apple Pay, Google Pay, BLIK) plus the phrase *"payment is charged automatically at the end of the ride"*. The query embedding aligns more with `refund` chunks (which contain *"refund / pay back / charged"*) than with brand lists.

This is a **corpus design issue**, not a model limitation: a chunk listing payment instruments does not embed near a query about the *act* of paying.

### 5. UK chunks have higher absolute similarities but worse recall

UK top-1 similarities (0.65–0.71 across all 5 queries) are higher than EN top-1s (0.39–0.60), yet UK recall@5 (60%) is the lowest. High similarity with the wrong chunk is a known symptom of underrepresented languages in the training corpus — the model produces "in-distribution" Ukrainian vectors that all look similar.

## Implications for NanoVec

1. **`all-MiniLM-L6-v2` is unsuitable for multilingual retrieval.** It is appropriate for English-only workloads or as a default that documents its own limitation.
2. **For real multilingual support**, swap to a model trained on parallel data:
   - `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, drop-in replacement)
   - `intfloat/multilingual-e5-base` (768-dim, higher quality)
   - `BAAI/bge-m3` (1024-dim, state-of-the-art multilingual + cross-lingual)
3. **Embedder swap is a configuration concern, not a NanoVec architecture concern.** The MCP layer, vector store, and distance functions are model-agnostic; only the embedder service binding needs to change. Dimension lock will reset on model swap (intentional safety).
4. **Document the limitation in `README.md`** so users know the default model is English-first, and that cross-lingual queries will return same-language results unless tokens like numbers or proper nouns provide an anchor.

## Reproducibility

Indexing and search were performed via the running NanoVec MCP server using `index_document` and `search_document` (cosine metric). Every chunk corresponds to one topic × one language section in `TEST-CORPUS-MULTILINGUAL.md`. The database was cleared at the end of the session via the `clear` MCP tool.

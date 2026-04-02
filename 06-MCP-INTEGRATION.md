# NanoVec: Model Context Protocol Integration

## MCP Overview

The **Model Context Protocol (MCP)** is an open standard that enables AI applications to securely connect to external data sources and tools. 
### Key Concepts

**MCP Architecture**:
```
┌──────────────────┐         ┌──────────────────┐
│   MCP Client     │         │   MCP Server     │
│  (Claude.app,    │◄───────►│  (NanoVec)       │
│   LangGraph)     │  JSON   │                  │
│                  │  -RPC   │  Exposes Tools:  │
│                  │  2.0    │  • index_doc     │
│                  │         │  • search        │
│                  │         │  • delete        │
└──────────────────┘         └──────────────────┘
```

**Benefits for NanoVec**:
- ✅ **Zero Integration Code**: No REST APIs or custom clients needed
- ✅ **Type-Safe**: JSON Schema prevents invalid queries
- ✅ **Discoverable**: LLMs query available tools automatically
- ✅ **Standardized**: Works with any MCP-compatible client

---

## Transport Layers

### 1. stdio Transport (Local Development)

**Use Case**: Single-machine, secure agent workflows.

**Architecture**:
```
┌─────────────────────────────────────┐
│      Claude Desktop / AI Client     │
│  ┌───────────────────────────────┐  │
│  │  MCP Client reads from:       │  │
│  │  - stdout (NanoVec responses) │  │
│  │  - stderr (logging/debug)     │  │
│  │                               │  │
│  │  MCP Client writes to:        │  │
│  │  - stdin (JSON-RPC requests)  │  │
│  └───────────────────────────────┘  │
└────────────┬────────────────────────┘
             │ Spawns subprocess
             ▼
┌─────────────────────────────────────┐
│     NanoVec Binary (stdio mode)     │
│  ┌───────────────────────────────┐  │
│  │  Reads JSON-RPC from stdin    │  │
│  │  Writes responses to stdout   │  │
│  │  Logs to stderr (CRITICAL!)   │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Implementation**:
```rust
use rmcp::server::stdio::StdioServer;
use rmcp::types::{ServerCapabilities, Tool, ToolInfo};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CRITICAL: Route all logging to stderr
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)  // NOT stdout!
        .init();
    
    let server = StdioServer::new(
        "nanovec",
        "0.1.0",
        ServerCapabilities {
            tools: true,
            resources: false,
            prompts: false,
        },
    );
    
    // Register tools
    server.add_tool(create_index_document_tool());
    server.add_tool(create_semantic_search_tool());
    server.add_tool(create_delete_document_tool());
    
    // Start listening on stdin/stdout
    server.run().await?;
    
    Ok(())
}
```

**Configuration** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "nanovec": {
      "command": "/path/to/nanovec",
      "args": ["--mode", "stdio"],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

**Critical stdio Rules**:
```rust
// ❌ NEVER do this in stdio mode
println!("Indexing document...");  // Corrupts stdout!

// ✅ ALWAYS route logs to stderr
eprintln!("Indexing document...");
tracing::info!("Indexed {} vectors", count);  // Goes to stderr
```

---

### 2. SSE Transport (Distributed Agents)

**Use Case**: Remote access, multi-agent swarms, cloud deployments.

**Architecture**:
```
┌────────────────────────────────────────┐
│    Remote MCP Client (Browser/App)     │
│         https://agent.example.com      │
└──────────────────┬─────────────────────┘
                   │ HTTPS + SSE
                   │ (Server-Sent Events)
                   ▼
┌────────────────────────────────────────┐
│        NanoVec SSE Server              │
│  ┌──────────────────────────────────┐  │
│  │  HTTP Endpoint: /sse              │  │
│  │  OAuth 2.1 Authentication         │  │
│  │  Rate Limiting (per-client)       │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

**Implementation**:
```rust
use axum::{Router, routing::get, extract::State};
use rmcp::server::sse::SseServer;
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp_server = create_mcp_server();
    
    let app = Router::new()
        .route("/sse", get(sse_handler))
        .layer(CorsLayer::permissive())
        .with_state(mcp_server);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn sse_handler(
    State(server): State<SseServer>,
) -> impl axum::response::IntoResponse {
    server.handle_connection().await
}
```

**Security Considerations**:
```rust
use oauth2::{AuthUrl, TokenUrl, basic::BasicClient};

/// OAuth 2.1 middleware
async fn require_auth(
    headers: axum::http::HeaderMap,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    let token = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Validate OAuth token
    validate_token(token).await?;
    
    Ok(next.run(req).await)
}
```

---

## Tool Definitions

### 1. index_document

**Purpose**: Chunk text, generate embeddings, insert vectors.

**JSON Schema**:
```json
{
  "name": "index_document",
  "description": "Index a text document by chunking, embedding, and storing in vector memory",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The document text to index"
      },
      "metadata": {
        "type": "object",
        "description": "Optional metadata (title, source, timestamp)",
        "properties": {
          "title": { "type": "string" },
          "source": { "type": "string" },
          "timestamp": { "type": "integer" }
        }
      },
      "chunk_size": {
        "type": "integer",
        "description": "Characters per chunk (default: 500)",
        "default": 500
      }
    },
    "required": ["text"]
  }
}
```

**Implementation**:
```rust
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct IndexDocumentInput {
    text: String,
    metadata: Option<serde_json::Value>,
    chunk_size: Option<usize>,
}

#[derive(Serialize)]
struct IndexDocumentOutput {
    document_id: u64,
    chunks_indexed: usize,
    total_vectors: usize,
}

async fn handle_index_document(
    input: IndexDocumentInput,
    db: &VectorDatabase,
    embedder: &EmbeddingModel,
) -> Result<IndexDocumentOutput, ToolError> {
    let chunk_size = input.chunk_size.unwrap_or(500);
    
    // 1. Chunk text
    let chunks = chunk_text(&input.text, chunk_size);
    
    // 2. Generate embeddings (batch call to embedding API)
    let embeddings = embedder.embed_batch(&chunks).await?;
    
    // 3. Insert vectors
    let mut indices = Vec::new();
    for (chunk_text, embedding) in chunks.iter().zip(embeddings.iter()) {
        let metadata = serde_json::json!({
            "text": chunk_text,
            "metadata": input.metadata,
        });
        
        let idx = db.insert(embedding.clone(), metadata.to_string())?;
        indices.push(idx);
    }
    
    // 4. Rebuild index if needed
    if db.should_rebuild_index() {
        db.rebuild_index();
    }
    
    Ok(IndexDocumentOutput {
        document_id: indices[0] as u64,
        chunks_indexed: chunks.len(),
        total_vectors: db.count(),
    })
}
```

**Text Chunking Strategy**:
```rust
/// Intelligent chunking with sentence boundaries
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    for sentence in sentences {
        if current_chunk.len() + sentence.len() > max_chars {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }
        }
        
        current_chunk.push_str(sentence);
        current_chunk.push(' ');
    }
    
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c| c == '.' || c == '?' || c == '!')
        .filter(|s| !s.trim().is_empty())
        .collect()
}
```

---

### 2. semantic_search

**Purpose**: Find K most similar vectors to query.

**JSON Schema**:
```json
{
  "name": "semantic_search",
  "description": "Search for semantically similar documents",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "k": {
        "type": "integer",
        "description": "Number of results to return (default: 5)",
        "default": 5,
        "minimum": 1,
        "maximum": 100
      },
      "min_score": {
        "type": "number",
        "description": "Minimum similarity score (0-1, default: 0.0)",
        "default": 0.0
      }
    },
    "required": ["query"]
  }
}
```

**Implementation**:
```rust
#[derive(Deserialize)]
struct SearchInput {
    query: String,
    k: Option<usize>,
    min_score: Option<f32>,
}

#[derive(Serialize)]
struct SearchOutput {
    results: Vec<SearchResult>,
    query_time_ms: f64,
}

#[derive(Serialize)]
struct SearchResult {
    id: u64,
    score: f32,
    text: String,
    metadata: serde_json::Value,
}

async fn handle_semantic_search(
    input: SearchInput,
    db: &VectorDatabase,
    embedder: &EmbeddingModel,
) -> Result<SearchOutput, ToolError> {
    let start = std::time::Instant::now();
    
    // 1. Embed query
    let query_embedding = embedder.embed(&input.query).await?;
    
    // 2. Search database
    let k = input.k.unwrap_or(5).min(100);
    let raw_results = db.search(&query_embedding, k)?;
    
    // 3. Filter by min_score if specified
    let min_score = input.min_score.unwrap_or(0.0);
    let filtered_results: Vec<_> = raw_results
        .into_iter()
        .filter(|(score, _)| *score >= min_score)
        .map(|(score, idx)| {
            let record = db.get_record(idx);
            SearchResult {
                id: record.id,
                score,
                text: extract_text_from_metadata(&record.metadata),
                metadata: serde_json::from_str(&record.metadata).unwrap_or_default(),
            }
        })
        .collect();
    
    let elapsed = start.elapsed();
    
    Ok(SearchOutput {
        results: filtered_results,
        query_time_ms: elapsed.as_secs_f64() * 1000.0,
    })
}
```

---

### 3. delete_document

**Purpose**: Remove vectors by ID or metadata filter.

**JSON Schema**:
```json
{
  "name": "delete_document",
  "description": "Delete documents from vector memory",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_id": {
        "type": "integer",
        "description": "Specific document ID to delete"
      },
      "metadata_filter": {
        "type": "object",
        "description": "Delete all documents matching metadata criteria"
      }
    }
  }
}
```

**Implementation**:
```rust
#[derive(Deserialize)]
struct DeleteInput {
    document_id: Option<u64>,
    metadata_filter: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct DeleteOutput {
    deleted_count: usize,
}

async fn handle_delete_document(
    input: DeleteInput,
    db: &mut VectorDatabase,
) -> Result<DeleteOutput, ToolError> {
    let deleted_count = if let Some(id) = input.document_id {
        // Delete specific ID
        db.delete_by_id(id)?;
        1
    } else if let Some(filter) = input.metadata_filter {
        // Delete by metadata filter
        db.delete_by_metadata_filter(&filter)?
    } else {
        return Err(ToolError::InvalidInput("Must provide document_id or metadata_filter"));
    };
    
    // Rebuild index after deletions
    db.rebuild_index();
    
    Ok(DeleteOutput { deleted_count })
}
```

---

## Integration with Embedding APIs

### OpenAI Embeddings

```rust
use reqwest::Client;

pub struct OpenAIEmbedder {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "text-embedding-3-small".to_string(),
        }
    }
    
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "input": text,
                "model": self.model,
            }))
            .send()
            .await?;
        
        let json: serde_json::Value = response.json().await?;
        
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or(EmbedError::InvalidResponse)?
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        
        Ok(embedding)
    }
    
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "input": texts,
                "model": self.model,
            }))
            .send()
            .await?;
        
        let json: serde_json::Value = response.json().await?;
        
        json["data"]
            .as_array()
            .ok_or(EmbedError::InvalidResponse)?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect()
    }
}
```

---

## Example Agent Workflow

### Scenario: Research Assistant

**User**: "Analyze this 100-page technical paper and answer questions about it."

**Agent Execution**:
```
1. Agent calls: index_document(text=paper_content, chunk_size=1000)
   └─> NanoVec: Chunks paper into 150 segments
       ├─> Calls OpenAI embedding API (batched)
       ├─> Inserts 150 vectors into memory
       └─> Returns: {document_id: 42, chunks_indexed: 150}

2. User asks: "What is the main contribution of Section 3?"
   
3. Agent calls: semantic_search(query="main contribution Section 3", k=5)
   └─> NanoVec: Embeds query, searches KD-Tree
       ├─> Finds top 5 relevant chunks (0.4ms latency)
       └─> Returns: [{score: 0.89, text: "Section 3 introduces..."}]

4. Agent synthesizes answer from retrieved chunks

5. Session ends
   └─> NanoVec process terminates → all vectors wiped from RAM
```

**MCP Message Flow**:
```json
// 1. Client → Server (index_document)
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "index_document",
    "arguments": {
      "text": "The paper discusses...",
      "chunk_size": 1000
    }
  }
}

// 2. Server → Client (response)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"document_id\": 42, \"chunks_indexed\": 150}"
    }]
  }
}

// 3. Client → Server (semantic_search)
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "semantic_search",
    "arguments": {
      "query": "main contribution Section 3",
      "k": 5
    }
  }
}

// 4. Server → Client (search results)
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"results\": [{\"score\": 0.89, \"text\": \"...\"}], \"query_time_ms\": 0.42}"
    }]
  }
}
```

---

## Deployment Strategies

### Local Development

```bash
# Build release binary
cargo build --release

# Configure Claude Desktop
cat > ~/Library/Application\ Support/Claude/claude_desktop_config.json <<EOF
{
  "mcpServers": {
    "nanovec": {
      "command": "./target/release/nanovec",
      "args": ["--stdio"]
    }
  }
}
EOF

# Restart Claude Desktop
killall Claude && open -a Claude
```

### Docker Deployment (SSE Mode)

```dockerfile
FROM rust:1.75-alpine AS builder

WORKDIR /build
COPY . .

RUN apk add --no-cache musl-dev && \
    cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/nanovec /nanovec

EXPOSE 8080
ENTRYPOINT ["/nanovec", "--sse", "--port", "8080"]
```

```bash
# Build and run
docker build -t nanovec:latest .
docker run -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY nanovec:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanovec
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nanovec
  template:
    metadata:
      labels:
        app: nanovec
    spec:
      containers:
      - name: nanovec
        image: nanovec:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            memory: "4Gi"  # Adjust based on vector dataset size
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: nanovec-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: nanovec
```

---

## Monitoring & Observability

### Structured Logging

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(db))]
async fn handle_search(
    query: SearchInput,
    db: &VectorDatabase,
) -> Result<SearchOutput, ToolError> {
    info!(
        query = %query.query,
        k = query.k.unwrap_or(5),
        "Received search request"
    );
    
    let start = std::time::Instant::now();
    let results = db.search(&query.query, query.k.unwrap_or(5))?;
    let elapsed = start.elapsed();
    
    info!(
        results_count = results.len(),
        query_time_ms = elapsed.as_millis(),
        "Search completed"
    );
    
    Ok(results)
}
```

### Metrics Collection

```rust
use prometheus::{register_histogram, register_counter, Histogram, IntCounter};

lazy_static! {
    static ref SEARCH_LATENCY: Histogram = register_histogram!(
        "nanovec_search_latency_seconds",
        "Search query latency"
    ).unwrap();
    
    static ref VECTORS_INDEXED: IntCounter = register_counter!(
        "nanovec_vectors_indexed_total",
        "Total vectors indexed"
    ).unwrap();
}

async fn handle_search_with_metrics(input: SearchInput) -> Result<SearchOutput, ToolError> {
    let timer = SEARCH_LATENCY.start_timer();
    let result = handle_search(input).await;
    timer.observe_duration();
    result
}
```

---

## Security Considerations

### API Key Management

```rust
use secrecy::{Secret, ExposeSecret};

pub struct SecureEmbedder {
    api_key: Secret<String>,
}

impl SecureEmbedder {
    pub fn new() -> Result<Self, EnvError> {
        let api_key = std::env::var("OPENAI_API_KEY")?;
        Ok(Self {
            api_key: Secret::new(api_key),
        })
    }
    
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // api_key.expose_secret() only when needed
        make_api_call(self.api_key.expose_secret(), text).await
    }
}
```

### Rate Limiting

```rust
use governor::{Quota, RateLimiter};

pub struct RateLimitedServer {
    limiter: RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>,
}

impl RateLimitedServer {
    pub fn new() -> Self {
        let quota = Quota::per_second(nonzero!(10u32));  // 10 requests/second
        Self {
            limiter: RateLimiter::keyed(quota),
        }
    }
    
    async fn check_rate_limit(&self, client_id: &str) -> Result<(), RateLimitError> {
        self.limiter.check_key(&client_id.to_string())?;
        Ok(())
    }
}
```

---

## Testing MCP Integration

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_index_and_search() {
        let db = VectorDatabase::new(768);
        let embedder = MockEmbedder::new();
        
        // Index document
        let index_result = handle_index_document(
            IndexDocumentInput {
                text: "The quick brown fox jumps over the lazy dog".to_string(),
                metadata: None,
                chunk_size: Some(20),
            },
            &db,
            &embedder,
        ).await.unwrap();
        
        assert!(index_result.chunks_indexed > 0);
        
        // Search
        let search_result = handle_semantic_search(
            SearchInput {
                query: "fox jumping".to_string(),
                k: Some(1),
                min_score: None,
            },
            &db,
            &embedder,
        ).await.unwrap();
        
        assert_eq!(search_result.results.len(), 1);
        assert!(search_result.results[0].score > 0.5);
    }
}
```

---

**Summary**: NanoVec's MCP integration provides a standardized, type-safe interface for AI agents to manage ephemeral vector memory without custom API development. The dual transport support (stdio for local, SSE for distributed) enables flexible deployment across use cases.


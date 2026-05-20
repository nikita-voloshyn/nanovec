//! Phase 6 T9 — end-to-end test of the streamable HTTP transport with two
//! logical clients, each scoped to its own `_conn_<id>` collection.
//!
//! Real wire test: spawns the actual axum-backed server on a random port,
//! uses reqwest to drive two distinct MCP sessions (each with its own
//! `connection_id`), indexes, searches, and verifies isolation.

use std::sync::Arc;

use nanovec::distance::Metric;
use nanovec::store::database::NanoVecDatabase;
use nanovec::transport::http;
use serde_json::{json, Value};

const DIM: usize = 8;

fn make_vec(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect()
}

async fn mcp_request(client: &reqwest::Client, url: &str, body: Value) -> Value {
    let resp = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(body.to_string())
        .send()
        .await
        .expect("send failed");
    assert!(resp.status().is_success(), "HTTP status {}", resp.status());
    let text = resp.text().await.expect("body read");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("invalid JSON `{text}`: {e}"))
}

async fn initialize(client: &reqwest::Client, url: &str) {
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "phase6-test", "version": "1.0"}
        }
    });
    let resp = mcp_request(client, url, init).await;
    assert_eq!(resp["jsonrpc"], "2.0", "init response: {resp}");
}

async fn call_tool(
    client: &reqwest::Client,
    url: &str,
    id: u64,
    tool: &str,
    arguments: Value,
) -> Value {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments}
    });
    mcp_request(client, url, body).await
}

/// Extract the JSON-decoded text payload from a `tools/call` response.
fn tool_text(resp: &Value) -> Value {
    let raw = resp
        .pointer("/result/content/0/text")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("missing content[0].text in {resp}"));
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("invalid JSON in tool text `{raw}`: {e}"))
}

#[tokio::test]
async fn two_http_clients_isolated_by_connection_id() -> anyhow::Result<()> {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 64));
    let (addr, ct, handle) = http::spawn_for_tests(Arc::clone(&db)).await?;
    let url = format!("http://{addr}/mcp");
    let client = reqwest::Client::new();

    initialize(&client, &url).await;

    // Alice indexes 3 vectors under connection_id="alice".
    for i in 0..3 {
        let resp = call_tool(
            &client,
            &url,
            100 + i,
            "index_vector",
            json!({
                "text": format!("alice-{i}"),
                "vector": make_vec(i as usize),
                "connection_id": "alice"
            }),
        )
        .await;
        let payload = tool_text(&resp);
        assert!(payload["id"].is_number(), "expected `id` in {payload}");
    }

    // Bob indexes 3 vectors under connection_id="bob".
    for i in 0..3 {
        let resp = call_tool(
            &client,
            &url,
            200 + i,
            "index_vector",
            json!({
                "text": format!("bob-{i}"),
                "vector": make_vec((i + 100) as usize),
                "connection_id": "bob"
            }),
        )
        .await;
        let payload = tool_text(&resp);
        assert!(payload["id"].is_number(), "expected `id` in {payload}");
    }

    // Server-side: confirm two _conn_* collections were auto-created.
    let snapshots = db.snapshot();
    let names: Vec<&str> = snapshots.iter().map(|s| s.name.as_str()).collect();
    println!("[http_e2e] collections after indexing: {names:?}");
    assert!(names.contains(&"_conn_alice"));
    assert!(names.contains(&"_conn_bob"));
    // Each should have exactly 3 records.
    for s in &snapshots {
        if s.name == "_conn_alice" || s.name == "_conn_bob" {
            assert_eq!(s.count, 3, "{} should have 3 records", s.name);
        }
    }

    // Alice searches → must only see her own data.
    let resp = call_tool(
        &client,
        &url,
        300,
        "search",
        json!({
            "vector": make_vec(0),
            "k": 10,
            "connection_id": "alice"
        }),
    )
    .await;
    let alice_hits = tool_text(&resp);
    let alice_results = alice_hits.as_array().expect("array");
    println!(
        "[http_e2e] alice search returned {} hits",
        alice_results.len()
    );
    assert_eq!(alice_results.len(), 3);
    for hit in alice_results {
        let text = hit["text"].as_str().unwrap();
        assert!(
            text.starts_with("alice-"),
            "alice should only see alice-*, got {text}"
        );
    }

    // Bob searches → must only see his own data.
    let resp = call_tool(
        &client,
        &url,
        400,
        "search",
        json!({
            "vector": make_vec(100),
            "k": 10,
            "connection_id": "bob"
        }),
    )
    .await;
    let bob_hits = tool_text(&resp);
    let bob_results = bob_hits.as_array().expect("array");
    println!("[http_e2e] bob search returned {} hits", bob_results.len());
    assert_eq!(bob_results.len(), 3);
    for hit in bob_results {
        let text = hit["text"].as_str().unwrap();
        assert!(
            text.starts_with("bob-"),
            "bob should only see bob-*, got {text}"
        );
    }

    // Cleanup.
    ct.cancel();
    let _ = handle.await;
    Ok(())
}

#[tokio::test]
async fn http_stats_tool_returns_memory_breakdown() -> anyhow::Result<()> {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 64));
    let (addr, ct, handle) = http::spawn_for_tests(Arc::clone(&db)).await?;
    let url = format!("http://{addr}/mcp");
    let client = reqwest::Client::new();

    initialize(&client, &url).await;

    for i in 0..50 {
        call_tool(
            &client,
            &url,
            i,
            "index_vector",
            json!({"text": format!("d-{i}"), "vector": make_vec(i as usize)}),
        )
        .await;
    }

    let resp = call_tool(&client, &url, 999, "stats", json!({})).await;
    let stats = tool_text(&resp);
    println!("[http_e2e] stats: {stats}");

    // New Phase 6 fields must be present.
    assert!(stats["memory"]["used_bytes"].as_u64().unwrap() > 0);
    assert!(stats["memory"]["limit_bytes"].as_u64().unwrap() > 0);
    assert!(stats["memory"]["collections_count"].as_u64().unwrap() >= 1);
    assert_eq!(stats["total_count"].as_u64().unwrap(), 50);

    // memory tool too.
    let resp = call_tool(&client, &url, 1000, "memory", json!({})).await;
    let mem = tool_text(&resp);
    println!("[http_e2e] memory: {mem}");
    assert!(mem["used_bytes"].as_u64().unwrap() > 0);
    assert!(mem["collections"].is_array());

    ct.cancel();
    let _ = handle.await;
    Ok(())
}

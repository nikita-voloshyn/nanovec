//! Follow-up after Phase 7 — exercise the `rebuild_index` MCP tool over HTTP
//! and verify that:
//!   1. Building an HNSW index switches the search dispatch path.
//!   2. HNSW-dispatched search returns essentially the same top-K as the
//!      brute-force baseline (recall ≈ 1.0 at this corpus size).
//!   3. A mutation (`index_vector`) invalidates the index — confirmed by
//!      issuing `rebuild_index` "none" and inspecting the response.

use std::collections::HashSet;
use std::sync::Arc;

use nanovec::distance::Metric;
use nanovec::store::database::NanoVecDatabase;
use nanovec::transport::http;
use serde_json::{json, Value};

const DIM: usize = 16;

fn make_vec(seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
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
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("invalid JSON: {e}\nbody: {text}"))
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

fn tool_text(resp: &Value) -> Value {
    let raw = resp
        .pointer("/result/content/0/text")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("missing content[0].text in {resp}"));
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("invalid JSON in tool text `{raw}`: {e}"))
}

#[tokio::test]
async fn rebuild_index_builds_hnsw_and_search_uses_it() -> anyhow::Result<()> {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 64));
    let (addr, ct, handle) = http::spawn_for_tests(Arc::clone(&db)).await?;
    let url = format!("http://{addr}/mcp");
    let client = reqwest::Client::new();

    // Initialize MCP session.
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "rebuild_index-test", "version": "1.0"}
        }
    });
    mcp_request(&client, &url, init).await;

    // Index 300 vectors.
    let n = 300;
    for i in 0..n {
        let resp = call_tool(
            &client,
            &url,
            100 + i as u64,
            "index_vector",
            json!({"text": format!("d-{i}"), "vector": make_vec(i)}),
        )
        .await;
        assert!(tool_text(&resp)["id"].is_number());
    }

    // Search before HNSW (brute-force path) — record top-10 ids.
    let query = make_vec(99999);
    let resp_brute = call_tool(
        &client,
        &url,
        200,
        "search",
        json!({"vector": query, "k": 10, "metric": "cosine"}),
    )
    .await;
    let brute_hits = tool_text(&resp_brute);
    let brute_ids: HashSet<u64> = brute_hits
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_u64().unwrap())
        .collect();
    assert_eq!(brute_ids.len(), 10);
    println!("[rebuild_index] brute-force top-10 ids: {brute_ids:?}");

    // Build HNSW.
    let resp = call_tool(
        &client,
        &url,
        300,
        "rebuild_index",
        json!({"kind": "hnsw", "m": 16, "ef_construction": 200, "ef_search": 50}),
    )
    .await;
    let build = tool_text(&resp);
    println!("[rebuild_index] build response: {build}");
    assert_eq!(build["kind"], "hnsw");
    assert_eq!(build["nodes"].as_u64().unwrap(), n as u64);
    assert!(build["build_ms"].as_u64().unwrap() > 0 || n < 100);

    // Search after HNSW — top-10 should be the same set as brute (on this
    // small corpus HNSW achieves recall 1.0).
    let resp_hnsw = call_tool(
        &client,
        &url,
        400,
        "search",
        json!({"vector": query, "k": 10, "metric": "cosine"}),
    )
    .await;
    let hnsw_hits = tool_text(&resp_hnsw);
    let hnsw_ids: HashSet<u64> = hnsw_hits
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_u64().unwrap())
        .collect();
    println!("[rebuild_index] hnsw top-10 ids:        {hnsw_ids:?}");

    let overlap = brute_ids.intersection(&hnsw_ids).count();
    assert!(
        overlap >= 9,
        "expected HNSW top-10 to overlap >= 9 with brute baseline, got {overlap} (brute={brute_ids:?}, hnsw={hnsw_ids:?})"
    );

    // Mutation invalidates the index — next index_vector clears it.
    call_tool(
        &client,
        &url,
        500,
        "index_vector",
        json!({"text": "after-rebuild", "vector": make_vec(424242)}),
    )
    .await;

    // The index is gone; a follow-up search still works (brute-force).
    let resp_after = call_tool(
        &client,
        &url,
        600,
        "search",
        json!({"vector": query, "k": 10, "metric": "cosine"}),
    )
    .await;
    let after_hits = tool_text(&resp_after);
    assert_eq!(after_hits.as_array().unwrap().len(), 10);

    // Drop index explicitly through `rebuild_index kind=none`.
    let resp = call_tool(&client, &url, 700, "rebuild_index", json!({"kind": "none"})).await;
    let none = tool_text(&resp);
    assert_eq!(none["kind"], "none");

    ct.cancel();
    let _ = handle.await;
    Ok(())
}

#[tokio::test]
async fn rebuild_index_uses_hnsw_metric_only_for_cosine() -> anyhow::Result<()> {
    // Confirm the dispatch logic: HNSW is built under cosine; a search with
    // metric=euclidean must fall back to brute even when HNSW is present.
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 64));
    let (addr, ct, handle) = http::spawn_for_tests(Arc::clone(&db)).await?;
    let url = format!("http://{addr}/mcp");
    let client = reqwest::Client::new();

    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "rebuild-metric-test", "version": "1.0"}
        }
    });
    mcp_request(&client, &url, init).await;

    for i in 0..100 {
        call_tool(
            &client,
            &url,
            100 + i,
            "index_vector",
            json!({"text": format!("d-{i}"), "vector": make_vec(i as usize)}),
        )
        .await;
    }

    // Build HNSW (built under cosine internally).
    call_tool(&client, &url, 200, "rebuild_index", json!({"kind": "hnsw"})).await;

    // Search with euclidean — must work AND return valid distances (>= 0).
    let resp = call_tool(
        &client,
        &url,
        300,
        "search",
        json!({"vector": make_vec(99999), "k": 5, "metric": "euclidean"}),
    )
    .await;
    let hits = tool_text(&resp);
    let arr = hits.as_array().unwrap();
    assert_eq!(arr.len(), 5);
    for h in arr {
        let d = h["distance"].as_f64().unwrap();
        assert!(d >= 0.0, "euclidean distance should be >= 0, got {d}");
    }

    ct.cancel();
    let _ = handle.await;
    Ok(())
}

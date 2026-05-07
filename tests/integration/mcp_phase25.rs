//! End-to-end checks for the Phase 2.5 MCP surface: `clear` tool, expanded
//! `stats` shape, and the new `distance`/`similarity` fields on search results.
//!
//! All assertions live in a single spawned binary to amortize the embedder
//! cold start across checks. Indexing uses `index_vector` (raw path) wherever
//! possible to keep tests fast and deterministic — `index_document` is only
//! used in passing where the document path is the actual subject under test.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn send_jsonrpc(stdin: &mut impl Write, msg: &str) {
    writeln!(stdin, "{msg}").unwrap();
    stdin.flush().unwrap();
}

fn read_jsonrpc(reader: &mut impl BufRead) -> String {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line.trim().to_string()
}

fn parse_tool_result(resp: &str, expected_id: u64) -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(resp).unwrap();
    assert_eq!(v["id"], expected_id, "response id mismatch (resp={resp})");
    let content = v["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("missing content text in response: {v}"));
    serde_json::from_str(content).unwrap_or_else(|e| panic!("content not JSON: {content} -> {e}"))
}

fn workspace_root() -> std::path::PathBuf {
    std::env::current_dir()
        .unwrap()
        .ancestors()
        .find(|p| p.join("Cargo.toml").exists())
        .unwrap_or(&std::env::current_dir().unwrap())
        .to_path_buf()
}

#[test]
fn phase_2_5_full_surface() {
    let root = workspace_root();
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(&root)
        .status()
        .expect("cargo build failed");
    assert!(status.success());

    let binary = root.join("target/debug/nanovec");
    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn nanovec");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // Build a 384-dim unit vector for raw-path indexing (no embedder needed
    // beyond the startup load).
    let dim = 384usize;
    let component = 1.0_f32 / (dim as f32).sqrt();
    let vec_str = (0..dim)
        .map(|_| format!("{component}"))
        .collect::<Vec<_>>()
        .join(",");

    // --- Initialize handshake -----------------------------------------------

    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}"#;
    send_jsonrpc(&mut stdin, init);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 1);

    let initialized = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    send_jsonrpc(&mut stdin, initialized);

    // --- Section A: clear ---------------------------------------------------
    //
    // Index 3 vectors, then `clear`. Verify the response, the post-clear
    // count, and that the next inserted document gets id=0 (next_id reset).

    let mut next_id = 2u64;
    let mut indexed_ids = vec![];
    for label in ["a", "b", "c"] {
        let req = format!(
            r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"{label}","vector":[{vec_str}]}}}}}}"#
        );
        send_jsonrpc(&mut stdin, &req);
        let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
        indexed_ids.push(result["id"].as_u64().unwrap());
        next_id += 1;
    }
    assert_eq!(indexed_ids.len(), 3);
    assert!(
        indexed_ids[0] < indexed_ids[1] && indexed_ids[1] < indexed_ids[2],
        "ids should be monotonically increasing before clear: {indexed_ids:?}",
    );

    let clear_req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"clear","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &clear_req);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(
        result["deleted"], 3,
        "clear should report pre-clear count: {result}"
    );

    // After clear, stats.count must be 0 and dimension must still be locked.
    let stats_req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"stats","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &stats_req);
    let stats = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(stats["count"], 0, "post-clear count should be 0: {stats}");

    // --- Section B: stats shape (Phase 2.5 expanded fields) -----------------

    assert_eq!(
        stats["dimension"], 384,
        "dimension still locked at 384: {stats}"
    );
    assert_eq!(
        stats["default_metric"]["raw_vector"], "euclidean",
        "raw-vector default should be euclidean: {stats}",
    );
    assert_eq!(
        stats["default_metric"]["document"], "cosine",
        "document default should be cosine: {stats}",
    );
    assert_eq!(
        stats["embedder"]["model"], "sentence-transformers/all-MiniLM-L6-v2",
        "embedder model identity: {stats}",
    );
    assert_eq!(
        stats["embedder"]["dim"], 384,
        "embedder dim mirrored in stats: {stats}",
    );
    // Backward-compat alias still present.
    assert!(
        stats["metric"].is_string(),
        "legacy `metric` alias missing: {stats}"
    );

    // --- Section C: id resets after clear ----------------------------------

    let after_clear_index = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"first after clear","vector":[{vec_str}]}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &after_clear_index);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(
        result["id"], 0,
        "after clear, next inserted document must get id=0: {result}",
    );

    // --- Section D: search returns distance + similarity (euclidean) -------
    //
    // The store currently has one vector identical to the query, so the
    // result distance should be ~0 and similarity = 1/(1+0) = 1.

    let search_req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":[{vec_str}],"k":1,"metric":"euclidean"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &search_req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;

    let arr = results.as_array().expect("search returns array");
    assert_eq!(arr.len(), 1, "expected 1 result: {results}");
    let r = &arr[0];

    assert_eq!(r["id"], 0);
    assert_eq!(r["text"], "first after clear");
    assert!(r["metadata"].is_object(), "metadata must be object: {r}");
    let distance = r["distance"]
        .as_f64()
        .unwrap_or_else(|| panic!("distance must be number: {r}"));
    let similarity = r["similarity"]
        .as_f64()
        .unwrap_or_else(|| panic!("similarity must be number: {r}"));
    let score = r["score"].as_f64().expect("score alias must be present");

    assert!(
        (distance - score).abs() < 1e-6,
        "score must alias distance: distance={distance} score={score}",
    );
    // similarity = 1 / (1 + distance) for euclidean.
    let expected_sim = 1.0 / (1.0 + distance);
    assert!(
        (similarity - expected_sim).abs() < 1e-5,
        "euclidean similarity mismatch: distance={distance} similarity={similarity} expected≈{expected_sim}",
    );
    // Self-match: distance ≈ 0, similarity ≈ 1.
    assert!(distance.abs() < 1e-3, "self-distance ≈ 0, got {distance}");
    assert!(
        (similarity - 1.0).abs() < 1e-3,
        "self-similarity ≈ 1, got {similarity}",
    );

    // --- Section E: search with cosine metric exposes 1-d similarity -------

    let search_cos = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":[{vec_str}],"k":1,"metric":"cosine"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &search_cos);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;

    let r = &results.as_array().unwrap()[0];
    let distance = r["distance"].as_f64().unwrap();
    let similarity = r["similarity"].as_f64().unwrap();
    let expected_sim = 1.0 - distance;
    assert!(
        (similarity - expected_sim).abs() < 1e-5,
        "cosine similarity = 1 - distance, got distance={distance} similarity={similarity}",
    );

    // --- Section F: empty clear is a noop ----------------------------------
    //
    // After clearing the single inserted vector, a second clear should report
    // deleted=0 without erroring.

    let clear2 = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"clear","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &clear2);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(result["deleted"], 1);

    let clear3 = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"clear","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &clear3);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    assert_eq!(
        result["deleted"], 0,
        "clearing an empty store should report 0: {result}",
    );

    child.kill().ok();
    child.wait().ok();
}

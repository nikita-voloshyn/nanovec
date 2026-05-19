//! Phase 4 end-to-end checks: multi-collection storage and metadata filtering
//! over the live MCP stdio transport. Mirrors the structure of
//! `mcp_phase25.rs` — one spawned binary, sequential tool calls, all asserts
//! against the JSON-RPC content text.

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
fn phase_4_collections_and_filtering() {
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

    // Two distinct 4-dim vectors used to keep the test fast (we sidestep the
    // embedder entirely by using `index_vector`). Collections in this test
    // are 4-dim, not 384-dim — the embedder dim only governs the default
    // collection's auto-creation.
    let v_a = "[1.0,0.0,0.0,0.0]";
    let v_b = "[0.0,1.0,0.0,0.0]";
    let v_c = "[0.0,0.0,1.0,0.0]";

    // ---- Handshake -----------------------------------------------------------
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}"#;
    send_jsonrpc(&mut stdin, init);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 1);
    send_jsonrpc(
        &mut stdin,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    );

    let mut next_id = 2u64;

    // ---- Section A: create two collections of different shape ----------------

    let create_docs = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"create_collection","arguments":{{"name":"docs","dimension":4}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &create_docs);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(result["created"], "docs");
    assert_eq!(result["dimension"], 4);

    let create_facts = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"create_collection","arguments":{{"name":"facts","dimension":4}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &create_facts);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(result["created"], "facts");

    // Duplicate creation must error.
    let create_dup = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"create_collection","arguments":{{"name":"docs","dimension":4}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &create_dup);
    let dup_resp = read_jsonrpc(&mut reader);
    let dup_v: serde_json::Value = serde_json::from_str(&dup_resp).unwrap();
    next_id += 1;
    // rmcp surfaces tool errors as `isError: true` content; we just need to
    // see SOMETHING shaped like an error response. The simplest robust check:
    // either there is an `error` field, or the content is flagged isError.
    let is_error = dup_v.get("error").is_some()
        || dup_v["result"]["isError"].as_bool().unwrap_or(false)
        || dup_v["result"]["content"][0]["text"]
            .as_str()
            .is_some_and(|s| s.contains("already exists"));
    assert!(
        is_error,
        "duplicate create_collection should surface an error: {dup_v}"
    );

    // ---- Section B: index documents into both collections --------------------

    // docs: two records, different `lang`.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"english doc","vector":{v_a},"metadata":{{"lang":"en"}},"collection":"docs"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let r1 = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert!(r1["id"].is_number());

    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"polish doc","vector":{v_b},"metadata":{{"lang":"pl"}},"collection":"docs"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;

    // facts: one record under a different namespace.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"isolated fact","vector":{v_c},"metadata":{{"lang":"en"}},"collection":"facts"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;

    // ---- Section C: search each collection with a filter ---------------------

    // docs filter lang=en — must NOT see "polish doc".
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":{v_a},"k":10,"collection":"docs","filter":{{"lang":"en"}}}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 1, "filter lang=en should match 1 doc: {results}");
    assert_eq!(arr[0]["text"], "english doc");

    // docs filter lang=pl — must see polish only.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":{v_b},"k":10,"collection":"docs","filter":{{"lang":"pl"}}}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["text"], "polish doc");

    // docs filter excluding everything (lang=ja) returns empty.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":{v_a},"k":10,"collection":"docs","filter":{{"lang":"ja"}}}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(results.as_array().unwrap().len(), 0);

    // docs search without filter returns both.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":{v_a},"k":10,"collection":"docs"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(results.as_array().unwrap().len(), 2);

    // facts search — collection isolation: must not see docs' records.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"search","arguments":{{"vector":{v_c},"k":10,"collection":"facts"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let results = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    let arr = results.as_array().unwrap();
    assert_eq!(arr.len(), 1, "facts is isolated from docs: {results}");
    assert_eq!(arr[0]["text"], "isolated fact");

    // ---- Section D: list_collections shape -----------------------------------

    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"list_collections","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    let collections = result["collections"].as_array().unwrap();
    let names: Vec<&str> = collections
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"docs"));
    assert!(names.contains(&"facts"));

    let docs_entry = collections
        .iter()
        .find(|c| c["name"] == "docs")
        .expect("docs entry");
    assert_eq!(docs_entry["count"], 2);
    assert_eq!(docs_entry["dimension"], 4);

    // ---- Section E: drop a collection and verify stats -----------------------

    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"drop_collection","arguments":{{"name":"facts"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(result["dropped"], "facts");

    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"stats","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let stats = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;

    // Backward-compat top-level fields still present.
    assert!(stats["count"].is_number(), "legacy `count` field missing");
    assert_eq!(
        stats["dimension"], 0,
        "default collection never materialized, so legacy dimension=0: {stats}"
    );
    assert!(stats["default_metric"]["raw_vector"].is_string());
    assert!(stats["embedder"]["model"].is_string());

    // Per-collection breakdown.
    let post_drop_collections = stats["collections"].as_array().unwrap();
    let post_drop_names: Vec<&str> = post_drop_collections
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(
        post_drop_names.contains(&"docs"),
        "docs should survive drop of facts: {stats}"
    );
    assert!(
        !post_drop_names.contains(&"facts"),
        "facts should be gone: {stats}"
    );

    let total_count = stats["total_count"].as_u64().unwrap();
    assert_eq!(total_count, 2, "only docs (2 records) remains: {stats}");

    // ---- Section F: legacy default-collection path still works ---------------

    // index_vector without a collection lands in `default`.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"legacy","vector":{v_a}}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let legacy_resp = read_jsonrpc(&mut reader);
    let legacy_v: serde_json::Value = serde_json::from_str(&legacy_resp).unwrap();
    next_id += 1;
    // The default collection is locked to the embedder dim (384), so a 4-dim
    // vector should be rejected — this is the documented behavior.
    let is_error = legacy_v.get("error").is_some()
        || legacy_v["result"]["isError"].as_bool().unwrap_or(false)
        || legacy_v["result"]["content"][0]["text"]
            .as_str()
            .is_some_and(|s| s.contains("dim mismatch") || s.contains("mismatch"));
    assert!(
        is_error,
        "legacy default with 4-dim should error since default is 384-dim: {legacy_v}"
    );

    // clear on default (with no records inserted because of dim mismatch)
    // still works without erroring.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"clear","arguments":{{}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    next_id += 1;
    assert_eq!(result["deleted"], 0);

    // clear specifying the docs collection explicitly.
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":{next_id},"method":"tools/call","params":{{"name":"clear","arguments":{{"collection":"docs"}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &req);
    let result = parse_tool_result(&read_jsonrpc(&mut reader), next_id);
    assert_eq!(result["deleted"], 2, "docs had 2 records before clear");

    child.kill().ok();
    child.wait().ok();
}

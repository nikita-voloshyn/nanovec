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

/// Find the workspace root (the directory that contains Cargo.toml).
fn workspace_root() -> std::path::PathBuf {
    std::env::current_dir()
        .unwrap()
        .ancestors()
        .find(|p| p.join("Cargo.toml").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap())
}

/// Build the nanovec binary and return its path.
fn build_and_locate_binary() -> std::path::PathBuf {
    let root = workspace_root();
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(&root)
        .status()
        .expect("cargo build failed");
    assert!(status.success(), "cargo build returned non-zero status");
    root.join("target/debug/nanovec")
}

/// Convenience: send a tools/call request with the given id, tool name, and
/// arguments JSON string, then read and parse the response.
fn call_tool(
    stdin: &mut impl Write,
    reader: &mut impl BufRead,
    id: u64,
    tool: &str,
    arguments: &str,
) -> serde_json::Value {
    let msg = format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"{tool}","arguments":{arguments}}}}}"#
    );
    send_jsonrpc(stdin, &msg);
    let raw = read_jsonrpc(reader);
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
        panic!("failed to parse JSON-RPC response for tool={tool}: {e}\nraw: {raw}")
    });
    assert_eq!(v["id"], id, "response id mismatch for tool={tool}: {v}");
    // Unwrap the content[0].text double-encoding used by rmcp.
    let text = v["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("missing result.content[0].text for tool={tool}: {v}"));
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("failed to parse inner JSON for tool={tool}: {e}\ntext: {text}"))
}

#[test]
fn test_semantic_search_end_to_end() {
    let binary = build_and_locate_binary();

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn nanovec");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // -------------------------------------------------------------------------
    // 1. Handshake: initialize + notifications/initialized + tools/list
    // -------------------------------------------------------------------------
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}"#;
    send_jsonrpc(&mut stdin, init);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 1, "initialize response id mismatch: {v}");

    let initialized = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    send_jsonrpc(&mut stdin, initialized);

    // Verify all 6 tools are registered.
    let list = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    send_jsonrpc(&mut stdin, list);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 2, "tools/list response id mismatch: {v}");
    let tools = v["result"]["tools"]
        .as_array()
        .expect("tools/list result.tools should be an array");
    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap_or(""))
        .collect();
    for expected in &[
        "index_vector",
        "index_document",
        "search",
        "search_document",
        "delete",
        "stats",
    ] {
        assert!(
            tool_names.contains(expected),
            "expected tool {expected} in tools/list, got: {tool_names:?}"
        );
    }

    // -------------------------------------------------------------------------
    // 2. Index 3 semantically distinct documents
    //    IDs are assigned sequentially starting at 0.
    // -------------------------------------------------------------------------
    let docs = [
        (
            "meeting with engineering team on friday at 3pm",
            r#"{"category":"calendar"}"#,
            "calendar",
        ),
        (
            "buy milk and bread on the way home",
            r#"{"category":"shopping"}"#,
            "shopping",
        ),
        (
            "git commit hash a3f9b2 broke the deploy pipeline",
            r#"{"category":"incident"}"#,
            "incident",
        ),
    ];

    let mut ids = [0u64; 3];
    for (i, (text, meta, _category)) in docs.iter().enumerate() {
        let arguments = format!(
            r#"{{"text":{},"metadata":{}}}"#,
            serde_json::to_string(text).unwrap(),
            meta,
        );
        let result = call_tool(
            &mut stdin,
            &mut reader,
            10 + i as u64,
            "index_document",
            &arguments,
        );
        assert!(
            result["id"].is_number(),
            "index_document should return {{\"id\": N}}, got: {result}"
        );
        ids[i] = result["id"].as_u64().unwrap();
        assert_eq!(
            ids[i], i as u64,
            "expected sequential id={i}, got ids[{i}]={}",
            ids[i]
        );
    }

    // -------------------------------------------------------------------------
    // 3. Three semantic searches — assert top-1 matches the obvious cluster
    // -------------------------------------------------------------------------

    // Helper: run search_document with k=3, return the result array.
    let semantic_search = |stdin: &mut _, reader: &mut _, req_id: u64, query: &str| {
        let arguments = format!(
            r#"{{"query":{},"k":3}}"#,
            serde_json::to_string(query).unwrap()
        );
        call_tool(stdin, reader, req_id, "search_document", &arguments)
    };

    // 3a. Calendar query → top-1 must be id=0
    let results = semantic_search(
        &mut stdin,
        &mut reader,
        20,
        "what meetings do I have this week?",
    );
    let arr = results
        .as_array()
        .expect("search_document should return an array");
    assert!(
        !arr.is_empty(),
        "search returned empty results for calendar query"
    );
    let top = &arr[0];
    assert_eq!(
        top["id"],
        ids[0],
        "calendar query: expected top-1 id={}, got id={}, scores: {:?}",
        ids[0],
        top["id"],
        arr.iter()
            .map(|r| (&r["id"], &r["score"]))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        top["metadata"]["category"], "calendar",
        "calendar query: expected metadata.category=calendar, got: {}",
        top["metadata"]
    );

    // 3b. Shopping query → top-1 must be id=1
    let results = semantic_search(&mut stdin, &mut reader, 21, "groceries to pick up");
    let arr = results
        .as_array()
        .expect("search_document should return an array");
    assert!(
        !arr.is_empty(),
        "search returned empty results for shopping query"
    );
    let top = &arr[0];
    assert_eq!(
        top["id"],
        ids[1],
        "shopping query: expected top-1 id={}, got id={}, scores: {:?}",
        ids[1],
        top["id"],
        arr.iter()
            .map(|r| (&r["id"], &r["score"]))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        top["metadata"]["category"], "shopping",
        "shopping query: expected metadata.category=shopping, got: {}",
        top["metadata"]
    );

    // 3c. Incident query → top-1 must be id=2
    let results = semantic_search(&mut stdin, &mut reader, 22, "what broke production?");
    let arr = results
        .as_array()
        .expect("search_document should return an array");
    assert!(
        !arr.is_empty(),
        "search returned empty results for incident query"
    );
    let top = &arr[0];
    assert_eq!(
        top["id"],
        ids[2],
        "incident query: expected top-1 id={}, got id={}, scores: {:?}",
        ids[2],
        top["id"],
        arr.iter()
            .map(|r| (&r["id"], &r["score"]))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        top["metadata"]["category"], "incident",
        "incident query: expected metadata.category=incident, got: {}",
        top["metadata"]
    );

    // -------------------------------------------------------------------------
    // 4. Delete id=1 (shopping doc), verify stats shows count=2, dimension=384
    // -------------------------------------------------------------------------
    let delete_args = format!(r#"{{"id":{}}}"#, ids[1]);
    let del_result = call_tool(&mut stdin, &mut reader, 30, "delete", &delete_args);
    assert_eq!(
        del_result["success"], true,
        "delete should return {{\"success\":true}}, got: {del_result}"
    );

    let stats = call_tool(&mut stdin, &mut reader, 31, "stats", "{}");
    assert_eq!(
        stats["count"], 2,
        "after deleting 1 of 3 docs, count should be 2: {stats}"
    );
    assert_eq!(
        stats["dimension"], 384,
        "dimension must remain 384 after delete: {stats}"
    );

    // -------------------------------------------------------------------------
    // 5. Re-run "groceries" query with k=3 — id=1 must NOT appear; length == 2
    // -------------------------------------------------------------------------
    let results = semantic_search(&mut stdin, &mut reader, 32, "groceries to pick up");
    let arr = results
        .as_array()
        .expect("post-delete search should return array");
    assert_eq!(
        arr.len(),
        2,
        "after delete only 2 docs remain, expected len=2 got len={}: {arr:?}",
        arr.len()
    );
    let result_ids: Vec<u64> = arr
        .iter()
        .map(|r| r["id"].as_u64().expect("result id should be u64"))
        .collect();
    assert!(
        !result_ids.contains(&ids[1]),
        "deleted id={} must not appear in post-delete results: {result_ids:?}",
        ids[1]
    );

    // -------------------------------------------------------------------------
    // 6. Teardown
    // -------------------------------------------------------------------------
    child.kill().ok();
    child.wait().ok();
}

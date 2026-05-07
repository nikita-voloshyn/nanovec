use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn send_jsonrpc(stdin: &mut impl Write, msg: &str) {
    // rmcp uses newline-delimited JSON (one JSON object per line)
    writeln!(stdin, "{msg}").unwrap();
    stdin.flush().unwrap();
}

fn read_jsonrpc(reader: &mut impl BufRead) -> String {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line.trim().to_string()
}

#[test]
fn test_mcp_full_flow() {
    // Build first
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(
            std::env::current_dir()
                .unwrap()
                .ancestors()
                .find(|p| p.join("Cargo.toml").exists())
                .unwrap_or(&std::env::current_dir().unwrap()),
        )
        .status()
        .expect("cargo build failed");
    assert!(status.success());

    // Locate binary relative to workspace root
    let binary = std::env::current_dir()
        .unwrap()
        .ancestors()
        .find(|p| p.join("Cargo.toml").exists())
        .unwrap_or(&std::env::current_dir().unwrap())
        .join("target/debug/nanovec");

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // suppress tracing output
        .spawn()
        .expect("failed to spawn nanovec");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // Build a 384-dim test vector. Each component is 1/sqrt(384) so the L2
    // norm equals 1. The embedder dim is locked at 384 from server startup,
    // so any other dim will be rejected by `index_vector`.
    let dim = 384usize;
    let component = 1.0_f32 / (dim as f32).sqrt();
    let vec_str = (0..dim)
        .map(|_| format!("{component}"))
        .collect::<Vec<_>>()
        .join(",");

    // 1. Initialize
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}"#;
    send_jsonrpc(&mut stdin, init);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 1, "initialize response id mismatch: {v}");

    // 2. Initialized notification (no response expected)
    let initialized = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    send_jsonrpc(&mut stdin, initialized);

    // 3. index_vector
    let index = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"index_vector","arguments":{{"text":"hello world","vector":[{vec_str}]}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &index);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 2, "index_vector response id mismatch: {v}");
    let content = &v["result"]["content"][0]["text"];
    let result: serde_json::Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert!(
        result["id"].is_number(),
        "Expected id in response: {result}"
    );
    let doc_id = result["id"].as_u64().unwrap();

    // 4. search
    let search = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"search","arguments":{{"vector":[{vec_str}],"k":1}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &search);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 3, "search response id mismatch: {v}");
    let content = &v["result"]["content"][0]["text"];
    let results: serde_json::Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert!(results.is_array(), "Expected array of results: {results}");
    assert_eq!(results.as_array().unwrap().len(), 1);
    assert_eq!(results[0]["id"], doc_id);
    assert_eq!(results[0]["text"], "hello world");

    // 5. delete
    let delete = format!(
        r#"{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"delete","arguments":{{"id":{doc_id}}}}}}}"#
    );
    send_jsonrpc(&mut stdin, &delete);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 4, "delete response id mismatch: {v}");

    // 6. stats — after delete, count should be 0
    let stats = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"stats","arguments":{}}}"#;
    send_jsonrpc(&mut stdin, stats);
    let resp = read_jsonrpc(&mut reader);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["id"], 5, "stats response id mismatch: {v}");
    let content = &v["result"]["content"][0]["text"];
    let s: serde_json::Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert_eq!(s["count"], 0, "After delete, count should be 0: {s}");
    // dimension is locked at 384 by the embedder and remains so even after delete.
    assert_eq!(
        s["dimension"], 384,
        "dimension should be locked at 384 by embedder: {s}"
    );

    // Kill the server
    child.kill().ok();
    child.wait().ok();
}

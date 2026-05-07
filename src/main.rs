#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // CRITICAL: all logging goes to stderr -- stdout is reserved for MCP JSON-RPC.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    nanovec::server::run().await
}

//! `flint` command line.

#[tokio::main]
async fn main() {
    if let Err(err) = flint_server::cli().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

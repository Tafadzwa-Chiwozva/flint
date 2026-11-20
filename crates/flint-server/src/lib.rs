//! HTTP server and generation CLI for Flint.
//!
//! The library owns the engine thread, the OpenAI routes, and the command
//! line. The `flint` binary is a thin wrapper around [`cli`].

#![deny(missing_docs)]

mod cli;
mod engine;
mod filter;
mod http;

pub use cli::cli;

use tokio::signal;

/// Bind the OpenAI server described by the process arguments in [`cli`].
///
/// This module keeps shutdown waiting next to the library so the binary stays
/// a one-liner and tests can build a router without installing a signal handler.
pub(crate) mod serve_runtime {
    use super::*;

    /// Resolves on ctrl-c.
    pub async fn shutdown_signal() {
        let _ = signal::ctrl_c().await;
        tracing::info!("shutdown");
    }
}

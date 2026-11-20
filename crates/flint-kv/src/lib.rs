//! Paged KV cache.
//!
//! Two pieces, split the way vLLM splits them:
//!
//! * [`BlockSpaceManager`] hands out fixed-size block ids, reference-counts
//!   them, and can fork a block table for prefix sharing. It does not store
//!   activations.
//! * [`PagedKvCache`] owns the key and value tensors. Attention reads them
//!   through a block table, so logical token order and physical storage order
//!   are independent.
//!
//! Prefix caching itself is phase 2. The refcount and copy-on-write API is
//! here, tested, and unused by the phase-1 scheduler on purpose.

#![deny(missing_docs)]

mod blocks;
mod cache;

pub use blocks::{BlockSpaceManager, CowBlock};
pub use cache::PagedKvCache;
pub use flint_kernels::{AttentionParams, CacheGeometry};

/// Errors from block allocation or cache indexing.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvError {
    /// Block id is outside this manager.
    #[error("block id {0} is out of range")]
    BadId(u32),
    /// Decrement or share of a block that is not live.
    #[error("block {0} has no live references")]
    Dangling(u32),
    /// Not enough free blocks for the request.
    #[error("out of KV blocks (need {need}, free {free})")]
    OutOfBlocks {
        /// Additional blocks that were requested.
        need: usize,
        /// Blocks on the free list at the time of the failure.
        free: usize,
    },
    /// A dimension was zero or the allocation overflowed.
    #[error("invalid cache geometry")]
    Geometry,
    /// A token slot or layer was outside the cache.
    #[error("cache index out of range")]
    Bounds,
    /// Refcount hit `u32::MAX`.
    #[error("block refcount overflow")]
    RefcountOverflow,
}

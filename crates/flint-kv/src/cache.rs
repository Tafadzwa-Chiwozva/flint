//! Key/value storage addressed by physical block id.

use flint_kernels::{paged_attention, AttentionParams, CacheGeometry};

use crate::blocks::{BlockSpaceManager, CowBlock};
use crate::KvError;

/// Paged key and value cache for every layer.
///
/// The scheduler never touches these bytes. It only hands the model a block
/// table. Writes land at `(block_table[pos / block_size], pos % block_size)`.
#[derive(Debug)]
pub struct PagedKvCache {
    geometry: CacheGeometry,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl PagedKvCache {
    /// Allocate a zeroed cache. The block ids are not reserved; pair this with
    /// a [`BlockSpaceManager`] of the same `num_blocks`.
    pub fn new(geometry: CacheGeometry) -> Result<Self, KvError> {
        if geometry.num_blocks == 0
            || geometry.num_layers == 0
            || geometry.num_kv_heads == 0
            || geometry.block_size == 0
            || geometry.head_dim == 0
        {
            return Err(KvError::Geometry);
        }
        let elems = geometry.elements().ok_or(KvError::Geometry)?;
        Ok(Self {
            geometry,
            keys: vec![0.0; elems],
            values: vec![0.0; elems],
        })
    }

    /// Convenience constructor used by the engine and tests.
    pub fn with_dims(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvError> {
        Self::new(CacheGeometry {
            num_blocks,
            num_layers,
            num_kv_heads,
            block_size,
            head_dim,
        })
    }

    /// Geometry shared with the attention kernel.
    #[must_use]
    pub fn geometry(&self) -> CacheGeometry {
        self.geometry
    }

    /// Tokens per block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.geometry.block_size
    }

    /// Layers stored in the cache.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.geometry.num_layers
    }

    /// KV heads stored per token.
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.geometry.num_kv_heads
    }

    /// Channels per head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.geometry.head_dim
    }

    /// Write one token's K and V at a physical slot.
    ///
    /// `k` and `v` are `[num_kv_heads, head_dim]` row-major. RoPE has already
    /// been applied to `k`. Values are stored unrotated.
    pub fn write_token(
        &mut self,
        layer: usize,
        block_id: u32,
        slot: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), KvError> {
        let kv = self.geometry.num_kv_heads * self.geometry.head_dim;
        if k.len() != kv || v.len() != kv {
            return Err(KvError::Bounds);
        }
        for head in 0..self.geometry.num_kv_heads {
            let src = head * self.geometry.head_dim;
            let dst = self.slot_offset(block_id, layer, head, slot)?;
            let end = dst + self.geometry.head_dim;
            self.keys[dst..end].copy_from_slice(&k[src..src + self.geometry.head_dim]);
            self.values[dst..end].copy_from_slice(&v[src..src + self.geometry.head_dim]);
        }
        Ok(())
    }

    /// Attention for one query token.
    ///
    /// `query` and `out` are `[num_qo_heads, head_dim]`. `params.seq_len` is the
    /// number of cached tokens visible to the query, including the one just written.
    pub fn attention(
        &self,
        query: &[f32],
        block_table: &[u32],
        params: AttentionParams,
        out: &mut [f32],
    ) -> Result<(), KvError> {
        if params.layer >= self.geometry.num_layers {
            return Err(KvError::Bounds);
        }
        paged_attention(
            out,
            query,
            &self.keys,
            &self.values,
            block_table,
            self.geometry,
            params,
        );
        Ok(())
    }

    /// Copy every layer and head of one block. Used by copy-on-write.
    pub fn copy_block(&mut self, src: u32, dst: u32) -> Result<(), KvError> {
        if src == dst {
            return Ok(());
        }
        let stride = self.geometry.block_stride();
        let src_off = self.block_offset(src)? * stride;
        let dst_off = self.block_offset(dst)? * stride;
        self.keys.copy_within(src_off..src_off + stride, dst_off);
        self.values.copy_within(src_off..src_off + stride, dst_off);
        Ok(())
    }

    /// Detach `id` if it is shared: copy bytes into a private block and drop
    /// the extra reference. Returns the id the caller should write from now on.
    pub fn cow_block(&mut self, mgr: &mut BlockSpaceManager, id: u32) -> Result<u32, KvError> {
        match mgr.cow_begin(id)? {
            CowBlock::Unique(id) => Ok(id),
            CowBlock::Shared { src, dst } => {
                self.copy_block(src, dst)?;
                mgr.cow_commit(src)?;
                Ok(dst)
            }
        }
    }

    fn block_offset(&self, block_id: u32) -> Result<usize, KvError> {
        let block = block_id as usize;
        if block >= self.geometry.num_blocks {
            return Err(KvError::Bounds);
        }
        Ok(block)
    }

    fn slot_offset(
        &self,
        block_id: u32,
        layer: usize,
        head: usize,
        slot: usize,
    ) -> Result<usize, KvError> {
        if layer >= self.geometry.num_layers
            || head >= self.geometry.num_kv_heads
            || slot >= self.geometry.block_size
        {
            return Err(KvError::Bounds);
        }
        let block = self.block_offset(block_id)?;
        Ok(self.geometry.offset(block, layer, head, slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_write_is_visible_to_attention() {
        let mut cache = PagedKvCache::with_dims(2, 4, 1, 1, 2).unwrap();
        // Two tokens in block 1 (not block 0), so a wrong identity mapping fails.
        cache
            .write_token(0, 1, 0, &[1.0, 0.0], &[4.0, 0.0])
            .unwrap();
        cache
            .write_token(0, 1, 1, &[0.0, 1.0], &[0.0, 5.0])
            .unwrap();
        let mut out = [0.0; 2];
        // Query aligned with the second key and large enough that softmax
        // collapses onto it. A reader that ignored the block table and used
        // block 0 would see zeros.
        cache
            .attention(
                &[0.0, 8.0],
                &[1],
                AttentionParams {
                    layer: 0,
                    seq_len: 2,
                    num_qo_heads: 1,
                    scale: 1.0,
                    sliding_window: None,
                },
                &mut out,
            )
            .unwrap();
        assert!(out[0] < 0.5, "unexpected value {out:?}");
        assert!(out[1] > 4.0, "unexpected value {out:?}");
    }

    #[test]
    fn cow_preserves_source_bytes() {
        let mut mgr = BlockSpaceManager::new(4, 4).unwrap();
        let mut cache = PagedKvCache::with_dims(4, 4, 1, 1, 2).unwrap();
        let id = mgr.alloc_one().unwrap();
        cache
            .write_token(0, id, 0, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        mgr.incref(id).unwrap();
        let private = cache.cow_block(&mut mgr, id).unwrap();
        assert_ne!(private, id);
        // Mutating the private copy must not change the shared original.
        cache
            .write_token(0, private, 0, &[9.0, 9.0], &[9.0, 9.0])
            .unwrap();
        let mut out = [0.0; 2];
        cache
            .attention(
                &[1.0, 0.0],
                &[id],
                AttentionParams {
                    layer: 0,
                    seq_len: 1,
                    num_qo_heads: 1,
                    scale: 1.0,
                    sliding_window: None,
                },
                &mut out,
            )
            .unwrap();
        // Only one token, value was [3, 4].
        assert!((out[0] - 3.0).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 4.0).abs() < 1e-5, "{out:?}");
        assert_eq!(mgr.refcount_of(id).unwrap(), 1);
        assert_eq!(mgr.refcount_of(private).unwrap(), 1);
    }
}

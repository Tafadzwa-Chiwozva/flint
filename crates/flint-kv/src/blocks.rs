//! Block-id allocator. Storage lives in [`crate::PagedKvCache`].

use crate::KvError;

/// Result of starting a copy-on-write on one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CowBlock {
    /// This sequence is the only owner. Keep writing to `id`.
    Unique(u32),
    /// `src` is shared. `dst` is a fresh block; copy bytes, then [`BlockSpaceManager::cow_commit`].
    Shared {
        /// Block that still holds the shared bytes.
        src: u32,
        /// Newly allocated private block.
        dst: u32,
    },
}

/// Free list plus a refcount per physical block.
///
/// A block on the free list has refcount 0. [`BlockSpaceManager::alloc_one`]
/// pops one and sets the refcount to 1. [`BlockSpaceManager::fork_table`]
/// increments, which is the primitive prefix sharing needs: two sequences
/// point at the same ids until one of them diverges and copy-on-writes.
#[derive(Debug)]
pub struct BlockSpaceManager {
    block_size: usize,
    refcount: Vec<u32>,
    free: Vec<u32>,
}

impl BlockSpaceManager {
    /// `num_blocks` physical blocks, each holding `block_size` tokens.
    ///
    /// The first allocation returns id 0.
    pub fn new(num_blocks: usize, block_size: usize) -> Result<Self, KvError> {
        if num_blocks == 0 || block_size == 0 {
            return Err(KvError::Geometry);
        }
        let mut free: Vec<u32> = (0..num_blocks as u32).collect();
        free.reverse();
        Ok(Self {
            block_size,
            refcount: vec![0; num_blocks],
            free,
        })
    }

    /// Tokens stored in one block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Physical blocks this manager was created with.
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.refcount.len()
    }

    /// Maximum tokens addressable if every block is used by one sequence.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_blocks() * self.block_size
    }

    /// Blocks with refcount 0.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Blocks required to hold `num_tokens` tokens.
    #[must_use]
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        if num_tokens == 0 {
            0
        } else {
            num_tokens.div_ceil(self.block_size)
        }
    }

    /// Current refcount, or an error if `id` is out of range.
    pub fn refcount_of(&self, id: u32) -> Result<u32, KvError> {
        self.refcount
            .get(id as usize)
            .copied()
            .ok_or(KvError::BadId(id))
    }

    /// Allocate one block and set its refcount to 1.
    pub fn alloc_one(&mut self) -> Result<u32, KvError> {
        let id = self
            .free
            .pop()
            .ok_or(KvError::OutOfBlocks { need: 1, free: 0 })?;
        self.refcount[id as usize] = 1;
        Ok(id)
    }

    /// Grow `table` so it can address `num_tokens` tokens.
    ///
    /// On failure the table is unchanged and no block is taken off the free list.
    pub fn reserve(&mut self, table: &mut Vec<u32>, num_tokens: usize) -> Result<(), KvError> {
        if num_tokens == 0 {
            return Ok(());
        }
        let need = self.blocks_for_tokens(num_tokens);
        if table.len() >= need {
            return Ok(());
        }
        let extra = need - table.len();
        if self.free.len() < extra {
            return Err(KvError::OutOfBlocks {
                need: extra,
                free: self.free.len(),
            });
        }
        for _ in 0..extra {
            table.push(self.alloc_one()?);
        }
        Ok(())
    }

    /// Increment the refcount of a live block.
    pub fn incref(&mut self, id: u32) -> Result<(), KvError> {
        let rc = self
            .refcount
            .get_mut(id as usize)
            .ok_or(KvError::BadId(id))?;
        if *rc == 0 {
            return Err(KvError::Dangling(id));
        }
        *rc = rc.checked_add(1).ok_or(KvError::RefcountOverflow)?;
        Ok(())
    }

    /// Decrement the refcount. The block returns to the free list at 0.
    pub fn decref(&mut self, id: u32) -> Result<(), KvError> {
        let rc = self
            .refcount
            .get_mut(id as usize)
            .ok_or(KvError::BadId(id))?;
        if *rc == 0 {
            return Err(KvError::Dangling(id));
        }
        *rc -= 1;
        if *rc == 0 {
            self.free.push(id);
        }
        Ok(())
    }

    /// Drop one reference for each id. Shared blocks stay live.
    pub fn free_ids(&mut self, ids: &[u32]) -> Result<(), KvError> {
        for &id in ids {
            self.decref(id)?;
        }
        Ok(())
    }

    /// Share an existing table. Each id's refcount increases by one.
    ///
    /// The returned table is the prefix-cache hit: the new sequence reads the
    /// same physical blocks until [`Self::cow_begin`] detaches one.
    pub fn fork_table(&mut self, table: &[u32]) -> Result<Vec<u32>, KvError> {
        for &id in table {
            if self.refcount_of(id)? == 0 {
                return Err(KvError::Dangling(id));
            }
        }
        for &id in table {
            self.incref(id)?;
        }
        Ok(table.to_vec())
    }

    /// Begin copy-on-write.
    ///
    /// If the block is shared, a new id is allocated but `src` is not released
    /// yet. Copy the bytes, then call [`Self::cow_commit`] so a failure in
    /// between cannot drop the last copy of the data.
    pub fn cow_begin(&mut self, id: u32) -> Result<CowBlock, KvError> {
        let rc = self.refcount_of(id)?;
        if rc == 0 {
            return Err(KvError::Dangling(id));
        }
        if rc == 1 {
            return Ok(CowBlock::Unique(id));
        }
        let dst = self.alloc_one()?;
        Ok(CowBlock::Shared { src: id, dst })
    }

    /// Release the shared reference after the caller has copied `src` to `dst`.
    pub fn cow_commit(&mut self, src: u32) -> Result<(), KvError> {
        self.decref(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_and_reuse() {
        let mut mgr = BlockSpaceManager::new(2, 4).unwrap();
        assert_eq!(mgr.free_count(), 2);
        let a = mgr.alloc_one().unwrap();
        let b = mgr.alloc_one().unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert!(mgr.alloc_one().is_err());
        mgr.decref(a).unwrap();
        assert_eq!(mgr.free_count(), 1);
        let c = mgr.alloc_one().unwrap();
        assert_eq!(c, a);
        assert_eq!(mgr.refcount_of(c).unwrap(), 1);
    }

    #[test]
    fn refcount_keeps_block_until_last_free() {
        let mut mgr = BlockSpaceManager::new(1, 8).unwrap();
        let id = mgr.alloc_one().unwrap();
        mgr.incref(id).unwrap();
        assert_eq!(mgr.refcount_of(id).unwrap(), 2);
        mgr.decref(id).unwrap();
        assert_eq!(mgr.free_count(), 0);
        assert_eq!(mgr.refcount_of(id).unwrap(), 1);
        mgr.decref(id).unwrap();
        assert_eq!(mgr.free_count(), 1);
        assert_eq!(mgr.refcount_of(id).unwrap(), 0);
        assert_eq!(mgr.decref(id), Err(KvError::Dangling(id)));
    }

    #[test]
    fn reserve_is_transactional() {
        let mut mgr = BlockSpaceManager::new(2, 4).unwrap();
        let mut table = Vec::new();
        mgr.reserve(&mut table, 4).unwrap();
        assert_eq!(table.len(), 1);
        let err = mgr.reserve(&mut table, 12).unwrap_err();
        assert_eq!(err, KvError::OutOfBlocks { need: 2, free: 1 });
        assert_eq!(table.len(), 1);
        assert_eq!(mgr.free_count(), 1);
    }

    #[test]
    fn fork_and_cow() {
        let mut mgr = BlockSpaceManager::new(4, 4).unwrap();
        let mut table = Vec::new();
        mgr.reserve(&mut table, 8).unwrap();
        assert_eq!(table.len(), 2);
        let forked = mgr.fork_table(&table).unwrap();
        assert_eq!(forked, table);
        assert_eq!(mgr.refcount_of(table[0]).unwrap(), 2);

        match mgr.cow_begin(table[0]).unwrap() {
            CowBlock::Shared { src, dst } => {
                assert_eq!(src, table[0]);
                assert_ne!(src, dst);
                assert_eq!(mgr.refcount_of(src).unwrap(), 2);
                assert_eq!(mgr.refcount_of(dst).unwrap(), 1);
                mgr.cow_commit(src).unwrap();
                assert_eq!(mgr.refcount_of(src).unwrap(), 1);
            }
            CowBlock::Unique(_) => panic!("expected a shared block"),
        }

        // table[0] was committed back to a single owner. table[1] is still shared.
        assert_eq!(mgr.cow_begin(table[0]).unwrap(), CowBlock::Unique(table[0]));
        assert!(matches!(
            mgr.cow_begin(table[1]).unwrap(),
            CowBlock::Shared { src, .. } if src == table[1]
        ));
    }

    #[test]
    fn rejects_zero_geometry() {
        assert_eq!(BlockSpaceManager::new(0, 4).unwrap_err(), KvError::Geometry);
        assert_eq!(BlockSpaceManager::new(4, 0).unwrap_err(), KvError::Geometry);
    }
}

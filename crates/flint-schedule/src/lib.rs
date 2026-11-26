//! Continuous-batching scheduler.
//!
//! The scheduler does not run the model. Each [`Scheduler::schedule`] call
//! returns a feasible iteration: which sequences to prefill or decode, the
//! tokens and block table to use, and which sequences were preempted or
//! failed. The engine executes that iteration and reports progress back with
//! [`Scheduler::mark_computed`] and [`Scheduler::push_token`].
//!
//! Policy, in order:
//!
//! 1. Decode steps of already-running sequences are scheduled before prefills.
//!    That keeps inter-token latency from being stalled by a new prompt.
//! 2. Prefills are chunked (`prefill_chunk_size`) and the whole iteration is
//!    capped by `max_batched_tokens`.
//! 3. A waiting request is admitted only when its **entire current token list**
//!    fits in free blocks, and only while `max_num_seqs` allows it. The queue
//!    is FCFS and does not skip the head, so a large prompt cannot be starved
//!    by a stream of tiny ones.
//! 4. If a running sequence needs a new block and the free list is empty, the
//!    newest other sequence is preempted. Preemption drops its KV blocks and
//!    sets `computed` back to 0 (recompute, not swap). It is not re-admitted
//!    in the same iteration.
//! 5. A sequence that can no longer fit even in an empty cache fails instead
//!    of wedging the queue.

#![deny(missing_docs)]

use std::collections::{HashMap, HashSet, VecDeque};

use flint_kv::BlockSpaceManager;

/// Knobs for one engine.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Running sequences at once.
    pub max_num_seqs: usize,
    /// Token budget of one iteration, counting prefill tokens and decode tokens.
    pub max_batched_tokens: usize,
    /// Maximum prompt tokens computed for one sequence in one iteration.
    pub prefill_chunk_size: usize,
    /// Waiting requests before [`Scheduler::add_request`] returns [`ScheduleError::Overloaded`].
    pub max_waiting: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 8,
            max_batched_tokens: 512,
            prefill_chunk_size: 128,
            max_waiting: 64,
        }
    }
}

/// Why a request was rejected before it ever ran.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScheduleError {
    /// Prompt had no tokens.
    #[error("prompt is empty")]
    EmptyPrompt,
    /// `max_tokens` was zero.
    #[error("max_tokens must be at least 1")]
    BadMaxTokens,
    /// Prompt cannot fit in the KV cache even with no other sequence resident.
    #[error("prompt of {tokens} tokens exceeds KV capacity of {capacity_tokens} tokens")]
    TooLong {
        /// Prompt length that was rejected.
        tokens: usize,
        /// `num_blocks * block_size`.
        capacity_tokens: usize,
    },
    /// Waiting queue is full.
    #[error("too many waiting requests")]
    Overloaded,
    /// `id` is already queued or running.
    #[error("duplicate request id {0}")]
    Duplicate(u64),
    /// The id is not known to the scheduler.
    #[error("unknown request id {0}")]
    Unknown(u64),
    /// Computed count would move past the tokens stored for the sequence.
    #[error("computed {computed} + {n} exceeds {len} tokens for request {id}")]
    BadAdvance {
        /// Request that was advanced.
        id: u64,
        /// Tokens already in the cache.
        computed: usize,
        /// Tokens the engine tried to mark.
        n: usize,
        /// Tokens currently stored.
        len: usize,
    },
    /// SchedulerConfig or block pool was unusable.
    #[error("{0}")]
    Config(&'static str),
    /// Lookahead could not reserve blocks. The sequence's table is unchanged.
    #[error("out of KV blocks (need {need}, free {free})")]
    OutOfBlocks {
        /// Additional blocks that were requested.
        need: usize,
        /// Blocks on the free list.
        free: usize,
    },
}

/// One sequence's work for this iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkItem {
    /// Caller-supplied request id.
    pub request_id: u64,
    /// Tokens to run, starting at [`Self::position`].
    pub tokens: Vec<u32>,
    /// Absolute position of `tokens[0]`.
    pub position: usize,
    /// Physical block ids covering the whole sequence, not just this chunk.
    pub block_table: Vec<u32>,
    /// When true, the last token of this chunk is the last token currently in
    /// the sequence, so the engine should sample a new token from its logits.
    pub sample_after: bool,
}

/// What happened to a sequence that could not be scheduled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedFailure {
    /// Request id.
    pub request_id: u64,
    /// Human-readable cause. Also forwarded to the client.
    pub reason: String,
    /// Original prompt length.
    pub prompt_tokens: usize,
    /// Tokens appended after the prompt.
    pub completion_tokens: usize,
}

/// Result of one scheduling iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedStep {
    /// Sequences whose KV was dropped. They are back on the waiting queue and
    /// will recompute when blocks free up.
    pub preempted: Vec<u64>,
    /// Sequences removed from the scheduler. The engine should surface `reason`.
    pub failed: Vec<SchedFailure>,
    /// Work to run, decodes first, then prefills. Oldest first within each class.
    pub items: Vec<WorkItem>,
}

/// Token counts captured when a sequence leaves the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqStats {
    /// Prompt length at admission.
    pub prompt_tokens: usize,
    /// Tokens pushed after the prompt.
    pub completion_tokens: usize,
}

/// Result of appending a sampled token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// The sequence still has budget and should be decoded next iteration.
    Continue,
    /// `max_tokens` was reached. Blocks have been freed.
    Finished(SeqStats),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Waiting,
    Running,
}

struct Sequence {
    tokens: Vec<u32>,
    prompt_len: usize,
    computed: usize,
    block_table: Vec<u32>,
    max_tokens: usize,
    status: Status,
}

/// Owns the waiting queue, the running set, and the block allocator.
pub struct Scheduler {
    cfg: SchedulerConfig,
    blocks: BlockSpaceManager,
    sequences: HashMap<u64, Sequence>,
    waiting: VecDeque<u64>,
    running: Vec<u64>,
}

impl Scheduler {
    /// Create a scheduler and its block pool.
    pub fn new(
        cfg: SchedulerConfig,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<Self, ScheduleError> {
        if cfg.max_num_seqs == 0
            || cfg.max_batched_tokens == 0
            || cfg.prefill_chunk_size == 0
            || cfg.max_waiting == 0
        {
            return Err(ScheduleError::Config("scheduler limits must be non-zero"));
        }
        let blocks = BlockSpaceManager::new(num_blocks, block_size)
            .map_err(|_| ScheduleError::Config("invalid block pool"))?;
        Ok(Self {
            cfg,
            blocks,
            sequences: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
        })
    }

    /// Block size of the underlying pool.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.blocks.block_size()
    }

    /// Free physical blocks.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.blocks.free_count()
    }

    /// Ids of running sequences, oldest first.
    #[must_use]
    pub fn running(&self) -> &[u64] {
        &self.running
    }

    /// Ids of waiting sequences, next-to-admit first.
    #[must_use]
    pub fn waiting(&self) -> &VecDeque<u64> {
        &self.waiting
    }

    /// Tokens already written to the KV cache.
    #[must_use]
    pub fn computed(&self, id: u64) -> Option<usize> {
        self.sequences.get(&id).map(|seq| seq.computed)
    }

    /// Tokens stored for the sequence, including generated ones.
    #[must_use]
    pub fn token_len(&self, id: u64) -> Option<usize> {
        self.sequences.get(&id).map(|seq| seq.tokens.len())
    }

    /// Physical blocks currently held.
    #[must_use]
    pub fn block_table_len(&self, id: u64) -> Option<usize> {
        self.sequences.get(&id).map(|seq| seq.block_table.len())
    }

    /// Block ids for `id`, if the sequence is still known.
    #[must_use]
    pub fn block_table(&self, id: u64) -> Option<&[u32]> {
        self.sequences
            .get(&id)
            .map(|seq| seq.block_table.as_slice())
    }

    /// Grow the block table so `num_tokens` positions are addressable.
    ///
    /// The token list is unchanged. Speculative verification uses this to
    /// cover draft tokens that are not committed yet. On failure the table is
    /// left as it was.
    pub fn ensure_blocks(&mut self, id: u64, num_tokens: usize) -> Result<(), ScheduleError> {
        if !self.sequences.contains_key(&id) {
            return Err(ScheduleError::Unknown(id));
        }
        let mut table = std::mem::take(&mut self.sequences.get_mut(&id).unwrap().block_table);
        let result = self.blocks.reserve(&mut table, num_tokens);
        self.sequences.get_mut(&id).unwrap().block_table = table;
        match result {
            Ok(()) => Ok(()),
            Err(flint_kv::KvError::OutOfBlocks { need, free }) => {
                Err(ScheduleError::OutOfBlocks { need, free })
            }
            Err(_) => Err(ScheduleError::Config("block reserve failed")),
        }
    }

    /// Free blocks the sequence no longer needs once `num_tokens` is the length
    /// that must stay addressable (including a not-yet-computed pending token).
    ///
    /// Returns how many blocks went back to the free list.
    pub fn truncate_blocks(&mut self, id: u64, num_tokens: usize) -> Result<usize, ScheduleError> {
        if !self.sequences.contains_key(&id) {
            return Err(ScheduleError::Unknown(id));
        }
        let mut table = std::mem::take(&mut self.sequences.get_mut(&id).unwrap().block_table);
        let freed = self
            .blocks
            .truncate(&mut table, num_tokens)
            .expect("block table ids are live");
        self.sequences.get_mut(&id).unwrap().block_table = table;
        Ok(freed)
    }

    /// Queue a prompt. Admission happens later, inside [`Self::schedule`].
    pub fn add_request(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_tokens: usize,
    ) -> Result<(), ScheduleError> {
        if prompt.is_empty() {
            return Err(ScheduleError::EmptyPrompt);
        }
        if max_tokens == 0 {
            return Err(ScheduleError::BadMaxTokens);
        }
        if self.sequences.contains_key(&id) {
            return Err(ScheduleError::Duplicate(id));
        }
        let need = self.blocks.blocks_for_tokens(prompt.len());
        if need > self.blocks.total_blocks() {
            return Err(ScheduleError::TooLong {
                tokens: prompt.len(),
                capacity_tokens: self.blocks.total_tokens(),
            });
        }
        if self.waiting.len() >= self.cfg.max_waiting {
            return Err(ScheduleError::Overloaded);
        }
        let prompt_len = prompt.len();
        self.sequences.insert(
            id,
            Sequence {
                tokens: prompt,
                prompt_len,
                computed: 0,
                block_table: Vec::new(),
                max_tokens,
                status: Status::Waiting,
            },
        );
        self.waiting.push_back(id);
        Ok(())
    }

    /// Plan the next iteration.
    pub fn schedule(&mut self) -> SchedStep {
        let mut failed = Vec::new();
        let mut preempted = Vec::new();
        self.fail_impossible(&mut failed);
        self.prepare_running(&mut preempted, &mut failed);
        let just_preempted: HashSet<u64> = preempted.iter().copied().collect();
        self.admit(&just_preempted);
        let items = self.build_items();
        SchedStep {
            preempted,
            failed,
            items,
        }
    }

    /// Record that the engine wrote `n` tokens of KV for `id`.
    pub fn mark_computed(&mut self, id: u64, n: usize) -> Result<(), ScheduleError> {
        let seq = self
            .sequences
            .get_mut(&id)
            .ok_or(ScheduleError::Unknown(id))?;
        if seq.computed + n > seq.tokens.len() {
            return Err(ScheduleError::BadAdvance {
                id,
                computed: seq.computed,
                n,
                len: seq.tokens.len(),
            });
        }
        seq.computed += n;
        Ok(())
    }

    /// Append a sampled token. Finishes the sequence when `max_tokens` is hit.
    pub fn push_token(&mut self, id: u64, token: u32) -> Result<PushOutcome, ScheduleError> {
        let done = {
            let seq = self
                .sequences
                .get_mut(&id)
                .ok_or(ScheduleError::Unknown(id))?;
            seq.tokens.push(token);
            seq.tokens.len() - seq.prompt_len >= seq.max_tokens
        };
        if done {
            let stats = self.finish(id)?;
            Ok(PushOutcome::Finished(stats))
        } else {
            Ok(PushOutcome::Continue)
        }
    }

    /// Free blocks and forget the sequence.
    pub fn finish(&mut self, id: u64) -> Result<SeqStats, ScheduleError> {
        let seq = self
            .sequences
            .remove(&id)
            .ok_or(ScheduleError::Unknown(id))?;
        self.blocks
            .free_ids(&seq.block_table)
            .expect("block table ids are live");
        self.running.retain(|running_id| *running_id != id);
        self.waiting.retain(|waiting_id| *waiting_id != id);
        Ok(SeqStats {
            prompt_tokens: seq.prompt_len,
            completion_tokens: seq.tokens.len() - seq.prompt_len,
        })
    }

    fn fail_impossible(&mut self, failed: &mut Vec<SchedFailure>) {
        let ids: Vec<u64> = self.waiting.iter().copied().collect();
        for id in ids {
            let tokens = self.sequences[&id].tokens.len();
            if self.blocks.blocks_for_tokens(tokens) > self.blocks.total_blocks() {
                self.fail(id, "sequence no longer fits in the KV cache".into(), failed);
            }
        }
    }

    fn prepare_running(&mut self, preempted: &mut Vec<u64>, failed: &mut Vec<SchedFailure>) {
        let ids = self.running.clone();
        for id in ids {
            if !self.running.contains(&id) {
                continue;
            }
            let need_tokens = self.sequences[&id].tokens.len();
            let have = self.sequences[&id].block_table.len();
            let need = self.blocks.blocks_for_tokens(need_tokens);
            if have >= need {
                continue;
            }
            let extra = need - have;
            if self.blocks.free_count() < extra {
                self.preempt_until(id, extra, preempted);
            }
            if self.blocks.free_count() < extra {
                let reason = if need > self.blocks.total_blocks() {
                    "sequence no longer fits in the KV cache"
                } else {
                    "out of KV blocks"
                };
                self.fail(id, reason.into(), failed);
                continue;
            }
            let mut table = std::mem::take(&mut self.sequences.get_mut(&id).unwrap().block_table);
            match self.blocks.reserve(&mut table, need_tokens) {
                Ok(()) => {
                    self.sequences.get_mut(&id).unwrap().block_table = table;
                }
                Err(err) => {
                    if let Some(seq) = self.sequences.get_mut(&id) {
                        seq.block_table = table;
                    }
                    self.fail(id, err.to_string(), failed);
                }
            }
        }
    }

    fn preempt_until(&mut self, protect: u64, need_free: usize, preempted: &mut Vec<u64>) {
        while self.blocks.free_count() < need_free {
            let victim = self.running.iter().rev().copied().find(|id| *id != protect);
            let Some(victim) = victim else {
                break;
            };
            self.preempt(victim);
            preempted.push(victim);
        }
    }

    fn preempt(&mut self, id: u64) {
        let table = {
            let seq = self.sequences.get_mut(&id).expect("running id exists");
            seq.computed = 0;
            seq.status = Status::Waiting;
            std::mem::take(&mut seq.block_table)
        };
        self.blocks
            .free_ids(&table)
            .expect("preempted blocks are live");
        self.running.retain(|running_id| *running_id != id);
        self.waiting.push_front(id);
    }

    fn fail(&mut self, id: u64, reason: String, failed: &mut Vec<SchedFailure>) {
        let Some(seq) = self.sequences.remove(&id) else {
            return;
        };
        let _ = self.blocks.free_ids(&seq.block_table);
        self.running.retain(|running_id| *running_id != id);
        self.waiting.retain(|waiting_id| *waiting_id != id);
        failed.push(SchedFailure {
            request_id: id,
            reason,
            prompt_tokens: seq.prompt_len,
            completion_tokens: seq.tokens.len() - seq.prompt_len,
        });
    }

    fn admit(&mut self, just_preempted: &HashSet<u64>) {
        while self.running.len() < self.cfg.max_num_seqs {
            let Some(&id) = self.waiting.front() else {
                break;
            };
            if just_preempted.contains(&id) {
                break;
            }
            let need_tokens = self.sequences[&id].tokens.len();
            let need = self.blocks.blocks_for_tokens(need_tokens);
            if self.blocks.free_count() < need {
                break;
            }
            self.waiting.pop_front();
            let mut table = Vec::new();
            self.blocks
                .reserve(&mut table, need_tokens)
                .expect("free count was checked");
            let seq = self.sequences.get_mut(&id).expect("waiting id exists");
            seq.block_table = table;
            seq.status = Status::Running;
            self.running.push(id);
        }
    }

    fn build_items(&self) -> Vec<WorkItem> {
        let mut budget = self.cfg.max_batched_tokens;
        let mut decode_ids = Vec::new();
        let mut prefill_ids = Vec::new();
        for &id in &self.running {
            let seq = &self.sequences[&id];
            let pending = seq.tokens.len().saturating_sub(seq.computed);
            if pending == 0 {
                continue;
            }
            if seq.computed >= seq.prompt_len && pending == 1 {
                decode_ids.push(id);
            } else {
                prefill_ids.push(id);
            }
        }
        let mut items = Vec::new();
        for id in decode_ids {
            if budget == 0 {
                break;
            }
            self.push_work(&mut items, id, 1, &mut budget);
        }
        for id in prefill_ids {
            if budget == 0 {
                break;
            }
            let n = budget.min(self.cfg.prefill_chunk_size);
            self.push_work(&mut items, id, n, &mut budget);
        }
        items
    }

    fn push_work(&self, items: &mut Vec<WorkItem>, id: u64, max_n: usize, budget: &mut usize) {
        let seq = &self.sequences[&id];
        let pending = seq.tokens.len() - seq.computed;
        let n = pending.min(max_n).min(*budget);
        if n == 0 {
            return;
        }
        let start = seq.computed;
        items.push(WorkItem {
            request_id: id,
            tokens: seq.tokens[start..start + n].to_vec(),
            position: start,
            block_table: seq.block_table.clone(),
            sample_after: start + n == seq.tokens.len(),
        });
        *budget -= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(
        blocks: usize,
        block_size: usize,
        chunk: usize,
        budget: usize,
        seqs: usize,
    ) -> Scheduler {
        Scheduler::new(
            SchedulerConfig {
                max_num_seqs: seqs,
                max_batched_tokens: budget,
                prefill_chunk_size: chunk,
                max_waiting: 8,
            },
            blocks,
            block_size,
        )
        .unwrap()
    }

    fn prompt(n: usize) -> Vec<u32> {
        vec![1; n]
    }

    fn apply(sched: &mut Scheduler, step: &SchedStep, token: u32) {
        assert!(step.failed.is_empty(), "{:?}", step.failed);
        for item in &step.items {
            sched
                .mark_computed(item.request_id, item.tokens.len())
                .unwrap();
            if item.sample_after {
                let _ = sched.push_token(item.request_id, token);
            }
        }
    }

    #[test]
    fn rejects_empty_and_oversized_prompts() {
        let mut sched = scheduler(2, 4, 8, 32, 4);
        assert_eq!(
            sched.add_request(1, vec![], 4).unwrap_err(),
            ScheduleError::EmptyPrompt
        );
        assert!(matches!(
            sched.add_request(1, prompt(9), 1).unwrap_err(),
            ScheduleError::TooLong { .. }
        ));
    }

    #[test]
    fn chunked_prefill_samples_only_on_the_last_chunk() {
        let mut sched = scheduler(8, 4, 4, 32, 4);
        sched.add_request(1, prompt(10), 2).unwrap();
        let step = sched.schedule();
        assert_eq!(step.items.len(), 1);
        assert_eq!(step.items[0].tokens.len(), 4);
        assert!(!step.items[0].sample_after);
        assert_eq!(step.items[0].position, 0);
        apply(&mut sched, &step, 7);

        let step = sched.schedule();
        assert_eq!(step.items[0].position, 4);
        assert_eq!(step.items[0].tokens.len(), 4);
        assert!(!step.items[0].sample_after);
        apply(&mut sched, &step, 7);

        let step = sched.schedule();
        assert_eq!(step.items[0].tokens.len(), 2);
        assert!(step.items[0].sample_after);
        apply(&mut sched, &step, 7);
        assert_eq!(sched.token_len(1), Some(11));
        assert_eq!(sched.computed(1), Some(10));
    }

    #[test]
    fn decodes_run_before_prefills_inside_the_budget() {
        let mut sched = scheduler(16, 4, 8, 4, 4);
        sched.add_request(1, prompt(4), 4).unwrap();
        let step = sched.schedule();
        apply(&mut sched, &step, 9);
        // Request 1 now has one uncomputed token (a decode). Request 2 is a prefill.
        sched.add_request(2, prompt(8), 2).unwrap();
        let step = sched.schedule();
        assert!(step.items.len() >= 2, "{:?}", step.items);
        assert_eq!(step.items[0].request_id, 1);
        assert_eq!(step.items[0].tokens.len(), 1);
        assert!(step.items[0].sample_after);
        assert_eq!(step.items[1].request_id, 2);
        // Budget is 4: decode used 1, prefill gets 3 rather than the full chunk of 8.
        assert_eq!(step.items[1].tokens.len(), 3);
        assert!(!step.items[1].sample_after);
    }

    #[test]
    fn head_of_line_blocks_smaller_waiting_requests() {
        let mut sched = scheduler(4, 4, 16, 64, 4);
        sched.add_request(1, prompt(12), 1).unwrap();
        let step = sched.schedule();
        assert_eq!(sched.running(), &[1]);
        assert_eq!(step.items[0].tokens.len(), 12);
        sched.add_request(2, prompt(8), 1).unwrap();
        sched.add_request(3, prompt(4), 1).unwrap();
        let _ = sched.schedule();
        assert_eq!(sched.running(), &[1]);
        assert_eq!(
            sched.waiting().iter().copied().collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn max_num_seqs_caps_admission() {
        let mut sched = scheduler(16, 4, 8, 64, 1);
        sched.add_request(1, prompt(4), 1).unwrap();
        sched.add_request(2, prompt(4), 1).unwrap();
        let _ = sched.schedule();
        assert_eq!(sched.running(), &[1]);
        assert_eq!(sched.waiting().front().copied(), Some(2));
    }

    #[test]
    fn preemption_frees_blocks_and_resets_computed() {
        // 4 blocks * 4 tokens. Two prompts of 8 fill the cache exactly.
        let mut sched = scheduler(4, 4, 8, 64, 4);
        sched.add_request(1, prompt(8), 4).unwrap();
        sched.add_request(2, prompt(8), 4).unwrap();
        let step = sched.schedule();
        assert_eq!(step.items.len(), 2);
        assert!(step.preempted.is_empty());
        apply(&mut sched, &step, 3);
        assert_eq!(sched.free_blocks(), 0);
        assert_eq!(sched.token_len(1), Some(9));
        assert_eq!(sched.token_len(2), Some(9));

        let step = sched.schedule();
        assert_eq!(step.preempted, vec![2]);
        assert!(step.failed.is_empty());
        assert_eq!(step.items.len(), 1);
        assert_eq!(step.items[0].request_id, 1);
        assert!(step.items[0].sample_after);
        assert_eq!(sched.computed(2), Some(0));
        assert_eq!(sched.block_table_len(2), Some(0));
        assert_eq!(sched.token_len(2), Some(9));
        assert!(sched.running().contains(&1));
        assert!(!sched.running().contains(&2));
        // Not re-admitted in the same iteration.
        assert!(!step.items.iter().any(|item| item.request_id == 2));
    }

    #[test]
    fn finish_returns_blocks_so_a_waiter_can_run() {
        let mut sched = scheduler(2, 4, 8, 32, 4);
        sched.add_request(1, prompt(8), 1).unwrap();
        sched.add_request(2, prompt(4), 1).unwrap();
        let step = sched.schedule();
        assert_eq!(sched.running(), &[1]);
        apply(&mut sched, &step, 5);
        // max_tokens is 1, so apply finished request 1.
        assert!(sched.computed(1).is_none());
        assert_eq!(sched.free_blocks(), 2);
        let step = sched.schedule();
        assert_eq!(sched.running(), &[2]);
        assert_eq!(step.items[0].request_id, 2);
    }

    #[test]
    fn sequence_that_outgrows_the_cache_fails() {
        // Capacity is 8 tokens. Prompt 4 can be admitted; the 9th token cannot.
        let mut sched = scheduler(2, 4, 8, 32, 2);
        sched.add_request(1, prompt(4), 10).unwrap();
        // Drive until the scheduler reports a failure.
        let mut failed = None;
        for _ in 0..16 {
            let step = sched.schedule();
            if !step.failed.is_empty() {
                failed = Some(step.failed);
                break;
            }
            if step.items.is_empty() {
                break;
            }
            apply(&mut sched, &step, 1);
        }
        let failed = failed.expect("sequence should fail once it needs a third block");
        assert_eq!(failed[0].request_id, 1);
        assert!(sched.computed(1).is_none());
        assert_eq!(sched.free_blocks(), 2);
    }

    #[test]
    fn token_budget_prefers_decodes_over_a_waiting_prefill() {
        let mut sched = scheduler(16, 4, 8, 2, 8);
        for id in 1..=3 {
            sched.add_request(id, prompt(1), 4).unwrap();
        }
        let step = sched.schedule();
        assert_eq!(step.items.len(), 2, "budget 2 admits two 1-token prefills");
        apply(&mut sched, &step, 9);
        let step = sched.schedule();
        assert_eq!(step.items.len(), 2);
        assert!(step
            .items
            .iter()
            .all(|item| item.request_id != 3 && item.tokens.len() == 1 && item.sample_after));
    }

    #[test]
    fn lookahead_blocks_come_back_on_truncate() {
        let mut sched = scheduler(8, 2, 8, 32, 2);
        sched.add_request(1, prompt(4), 4).unwrap();
        let step = sched.schedule();
        assert_eq!(sched.block_table_len(1), Some(2));
        sched.ensure_blocks(1, 4 + 6).unwrap();
        assert_eq!(sched.block_table_len(1), Some(5));
        let freed = sched.truncate_blocks(1, 5).unwrap();
        assert_eq!(freed, 2);
        assert_eq!(sched.block_table_len(1), Some(3));
        assert_eq!(sched.free_blocks() + sched.block_table_len(1).unwrap(), 8);
        // The reserved tail was not committed, so the token list is unchanged.
        assert_eq!(sched.token_len(1), Some(4));
        let _ = step;
    }
}

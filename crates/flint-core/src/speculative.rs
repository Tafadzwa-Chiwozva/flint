//! Lossless draft-model speculative decoding.
//!
//! The draft model proposes `K` tokens from its own paged cache. The target
//! model scores those tokens in one multi-token forward and keeps the longest
//! prefix that its own distribution accepts.
//!
//! Greedy acceptance (`temperature == 0`) keeps a draft token only while it
//! equals `argmax` of the target logits at that position, then emits the
//! target argmax at the first mismatch, or the bonus argmax when all `K`
//! match. That sequence is the same one standard greedy decoding would have
//! produced. Stochastic acceptance is the Leviathan / Chen rejection sampler:
//! accept with probability `min(1, p/q)`, and on the first rejection draw from
//! `normalize(max(0, p - q))`. The bonus token, when every draft token is
//! kept, is drawn from the target distribution at position `K`.
//!
//! Both caches are truncated to the tokens that were actually committed. The
//! recovered token (the correction or the bonus) is left pending, the same way
//! a normal decode leaves the newly sampled token out of the KV cache until
//! the next step. Freed block ids go back to the free list.

use flint_kv::{BlockSpaceManager, CacheGeometry, PagedKvCache};
use rand::Rng;
#[cfg(test)]
use rand::SeedableRng;

use crate::error::Error;
use crate::model::Model;
use crate::sampling::{checked_argmax, sample_from_probs, sampling_distribution, SamplingParams};

/// Counters for one request, or for a whole speculative run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Tokens the draft model proposed.
    pub drafted_tokens: u64,
    /// Draft tokens the target model kept. The bonus or correction is not included.
    pub accepted_tokens: u64,
    /// Speculative steps that proposed at least one token.
    pub steps: u64,
}

impl SpeculativeStats {
    /// Record one speculative step.
    pub fn record(&mut self, drafted: u64, accepted: u64) {
        self.drafted_tokens = self.drafted_tokens.saturating_add(drafted);
        self.accepted_tokens = self.accepted_tokens.saturating_add(accepted);
        self.steps = self.steps.saturating_add(1);
    }

    /// `accepted / drafted`, or `0` before the first drafted token.
    #[must_use]
    pub fn acceptance_rate(self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.drafted_tokens as f64
        }
    }

    /// Mean number of draft tokens kept on each speculative step.
    #[must_use]
    pub fn mean_accepted_per_step(self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.steps as f64
        }
    }
}

/// How the draft model should produce the first of the `K` tokens.
pub enum DraftStart {
    /// The prefix is already in the draft cache. Sample from these logits.
    Logits(Vec<f32>),
    /// This token has been committed but not written. Forward it, then sample.
    Pending(u32),
}

/// Draft tokens and, for stochastic sampling, the distribution each one came from.
pub struct Proposal {
    /// `K` candidate token ids.
    pub tokens: Vec<u32>,
    /// One probability vector per drafted token. Empty when sampling is greedy.
    pub probs: Vec<Vec<f32>>,
}

/// What the target model kept from one proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptDecision {
    /// Draft tokens kept from the front of the proposal. `0..=K`.
    pub accepted: usize,
    /// Target token at the first rejection, or the bonus token when `accepted == K`.
    pub token: u32,
}

impl AcceptDecision {
    /// Accepted draft tokens followed by the recovered token.
    pub fn committed(&self, draft: &[u32]) -> Result<Vec<u32>, Error> {
        if self.accepted > draft.len() {
            return Err(Error::Inference(
                "speculative acceptance ran past the draft proposal".into(),
            ));
        }
        let mut out = draft[..self.accepted].to_vec();
        out.push(self.token);
        Ok(out)
    }
}

/// Tokens the target model must score, and how far its cache may roll back.
pub struct VerifyPlan {
    /// Chunk passed to the multi-token forward.
    pub tokens: Vec<u32>,
    /// Absolute position of [`Self::tokens`]`[0]`.
    pub position: usize,
    /// Logits already computed for the prefix. Prepended to the forward output
    /// so the acceptance routine always sees `K + 1` vectors.
    pub prefix_logits: Option<Vec<f32>>,
    /// Cache length that remains when every draft token is rejected.
    ///
    /// This includes a pending token that the forward writes first, and it
    /// excludes every draft token.
    pub base_len: usize,
}

/// Build the target verification chunk.
///
/// `pending` is the committed token that is not in the cache yet. The forward
/// is then `[pending] + draft` and returns `K + 1` logit rows directly.
/// Without a pending token the prefix logits (from prefill, or from the
/// previous bonus) are prepended to the `K` rows the forward returns for the
/// draft tokens alone.
pub fn plan_verify(
    kv_len: usize,
    pending: Option<u32>,
    prefix_logits: Option<Vec<f32>>,
    draft: &[u32],
) -> Result<VerifyPlan, Error> {
    if draft.is_empty() {
        return Err(Error::Inference(
            "speculative verification needs at least one draft token".into(),
        ));
    }
    match pending {
        Some(token) => {
            if prefix_logits.is_some() {
                return Err(Error::Inference(
                    "speculative verify was given both a pending token and prefix logits".into(),
                ));
            }
            let mut tokens = Vec::with_capacity(draft.len() + 1);
            tokens.push(token);
            tokens.extend_from_slice(draft);
            Ok(VerifyPlan {
                tokens,
                position: kv_len,
                prefix_logits: None,
                base_len: kv_len + 1,
            })
        }
        None => {
            if prefix_logits.is_none() {
                return Err(Error::Inference(
                    "speculative verify needs prefix logits when no token is pending".into(),
                ));
            }
            Ok(VerifyPlan {
                tokens: draft.to_vec(),
                position: kv_len,
                prefix_logits,
                base_len: kv_len,
            })
        }
    }
}

/// KV length to keep after acceptance. The recovered token is not included.
#[must_use]
pub fn keep_kv_len(base_len: usize, accepted: usize) -> usize {
    base_len + accepted
}

/// Refuse a draft whose vocabulary width differs from the target.
pub fn ensure_draft_compatible(target: &Model, draft: &Model) -> Result<(), Error> {
    let target_vocab = target.config().vocab_size;
    let draft_vocab = draft.config().vocab_size;
    if target_vocab == 0 || target_vocab != draft_vocab {
        return Err(Error::Config(format!(
            "draft vocab size {draft_vocab} is not compatible with target vocab size {target_vocab}"
        )));
    }
    Ok(())
}

/// A model that can score one chunk against a paged cache.
pub trait SpecModel {
    /// Logit width.
    fn vocab_size(&self) -> usize;
    /// Host or device cache geometry for a pool of `num_blocks`.
    fn cache_geometry(&self, num_blocks: usize, block_size: usize) -> CacheGeometry;
    /// Run `tokens` at `position` and return one logit vector per token when
    /// `all_positions` is set, otherwise a single vector for the last token.
    fn forward_spec(
        &mut self,
        cache: &mut PagedKvCache,
        tokens: &[u32],
        position: usize,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error>;
}

impl SpecModel for Model {
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn cache_geometry(&self, num_blocks: usize, block_size: usize) -> CacheGeometry {
        CacheGeometry {
            num_blocks,
            num_layers: self.config().num_hidden_layers,
            num_kv_heads: self.config().num_key_value_heads,
            block_size,
            head_dim: self.config().head_dim(),
        }
    }

    fn forward_spec(
        &mut self,
        cache: &mut PagedKvCache,
        tokens: &[u32],
        position: usize,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.forward_token_logits(tokens, position, cache, block_table, all_positions)
    }
}

#[cfg(feature = "cuda")]
impl SpecModel for crate::gpu::GpuModel {
    fn vocab_size(&self) -> usize {
        crate::gpu::GpuModel::vocab_size(self)
    }

    fn cache_geometry(&self, num_blocks: usize, block_size: usize) -> CacheGeometry {
        CacheGeometry {
            num_blocks,
            num_layers: self.num_layers(),
            num_kv_heads: self.num_kv_heads(),
            block_size,
            head_dim: self.head_dim(),
        }
    }

    fn forward_spec(
        &mut self,
        _cache: &mut PagedKvCache,
        tokens: &[u32],
        position: usize,
        block_table: &[u32],
        all_positions: bool,
    ) -> Result<Vec<Vec<f32>>, Error> {
        // The device cache is addressed by the same block table. The host
        // buffer the CPU oracle uses is not read.
        self.forward_token_logits(tokens, position, block_table, all_positions)
    }
}

/// Propose `k` tokens, writing each one into `cache` through `table`.
///
/// On return, `kv_len` includes every token the draft wrote: the pending token
/// when [`DraftStart::Pending`] was used, plus all `k` proposals. The caller
/// truncates both to the accepted prefix.
#[allow(clippy::too_many_arguments)]
pub fn draft_propose(
    model: &mut impl SpecModel,
    cache: &mut PagedKvCache,
    blocks: &mut BlockSpaceManager,
    table: &mut Vec<u32>,
    kv_len: &mut usize,
    start: DraftStart,
    k: usize,
    params: &SamplingParams,
    rng: &mut impl Rng,
) -> Result<Proposal, Error> {
    if k == 0 {
        return Err(Error::Inference(
            "speculative width must be at least 1".into(),
        ));
    }
    let mut next_logits = match start {
        DraftStart::Logits(logits) => logits,
        DraftStart::Pending(token) => forward_one(model, cache, blocks, table, kv_len, token)?,
    };
    let greedy = params.temperature == 0.0;
    let mut tokens = Vec::with_capacity(k);
    let mut probs = Vec::new();
    for _ in 0..k {
        let token = if greedy {
            checked_argmax(&next_logits)? as u32
        } else {
            let dist = sampling_distribution(&next_logits, params)?;
            let token = sample_from_probs(&dist, rng)?;
            probs.push(dist);
            token
        };
        tokens.push(token);
        next_logits = forward_one(model, cache, blocks, table, kv_len, token)?;
    }
    Ok(Proposal { tokens, probs })
}

fn forward_one(
    model: &mut impl SpecModel,
    cache: &mut PagedKvCache,
    blocks: &mut BlockSpaceManager,
    table: &mut Vec<u32>,
    kv_len: &mut usize,
    token: u32,
) -> Result<Vec<f32>, Error> {
    let position = *kv_len;
    blocks
        .reserve(table, position + 1)
        .map_err(|err| Error::Inference(err.to_string()))?;
    let mut rows = model.forward_spec(cache, &[token], position, table, false)?;
    *kv_len = position + 1;
    rows.pop()
        .filter(|row| !row.is_empty())
        .ok_or_else(|| Error::Inference("draft forward returned no logits".into()))
}

/// Greedy acceptance. `target_logits` has length `draft_tokens.len() + 1`.
pub fn accept_greedy(
    target_logits: &[Vec<f32>],
    draft_tokens: &[u32],
) -> Result<AcceptDecision, Error> {
    if target_logits.len() != draft_tokens.len() + 1 {
        return Err(Error::Inference(format!(
            "greedy verify expected {} logit vectors for {} draft tokens, got {}",
            draft_tokens.len() + 1,
            draft_tokens.len(),
            target_logits.len()
        )));
    }
    for (index, &token) in draft_tokens.iter().enumerate() {
        let choice = checked_argmax(&target_logits[index])? as u32;
        if choice != token {
            return Ok(AcceptDecision {
                accepted: index,
                token: choice,
            });
        }
    }
    let bonus = checked_argmax(target_logits.last().expect("length checked"))? as u32;
    Ok(AcceptDecision {
        accepted: draft_tokens.len(),
        token: bonus,
    })
}

/// Stochastic rejection sampling. Draft probabilities are the distributions the
/// proposal was drawn from. Target logits are converted with `params`, which
/// must be the same temperature, top-k, and top-p used for the draft.
pub fn accept_stochastic(
    target_logits: &[Vec<f32>],
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    params: &SamplingParams,
    rng: &mut impl Rng,
) -> Result<AcceptDecision, Error> {
    if params.temperature == 0.0 {
        return accept_greedy(target_logits, draft_tokens);
    }
    if draft_probs.len() != draft_tokens.len() {
        return Err(Error::Inference(
            "stochastic verify needs one draft distribution per proposed token".into(),
        ));
    }
    let mut target_probs = Vec::with_capacity(target_logits.len());
    for row in target_logits {
        target_probs.push(sampling_distribution(row, params)?);
    }
    reject_sample(&target_probs, draft_tokens, draft_probs, rng)
}

/// Rejection sampler on probability vectors. `target_probs` has length `K + 1`.
pub fn reject_sample(
    target_probs: &[Vec<f32>],
    draft_tokens: &[u32],
    draft_probs: &[Vec<f32>],
    rng: &mut impl Rng,
) -> Result<AcceptDecision, Error> {
    let k = draft_tokens.len();
    if k == 0 || target_probs.len() != k + 1 || draft_probs.len() != k {
        return Err(Error::Inference(
            "rejection sampling needs K draft tokens, K draft distributions, and K + 1 target distributions"
                .into(),
        ));
    }
    let vocab = target_probs[0].len();
    if vocab == 0 {
        return Err(Error::Inference(
            "rejection sampling was given an empty distribution".into(),
        ));
    }
    for (index, row) in target_probs.iter().enumerate() {
        if row.len() != vocab || row.iter().any(|prob| !prob.is_finite() || *prob < 0.0) {
            return Err(Error::Inference(format!(
                "target distribution {index} is not a valid probability vector"
            )));
        }
        if index < k
            && (draft_probs[index].len() != vocab
                || draft_probs[index]
                    .iter()
                    .any(|prob| !prob.is_finite() || *prob < 0.0))
        {
            return Err(Error::Inference(format!(
                "draft distribution {index} is not a valid probability vector"
            )));
        }
    }
    for (index, &token) in draft_tokens.iter().enumerate() {
        let id = token as usize;
        if id >= vocab {
            return Err(Error::Inference(format!(
                "draft token {token} is outside vocab {vocab}"
            )));
        }
        let p = target_probs[index][id];
        let q = draft_probs[index][id];
        let accept_prob = if q <= 0.0 {
            if p > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            (p / q).min(1.0)
        };
        if rng.gen::<f32>() < accept_prob {
            continue;
        }
        let recovered = sample_residual(&target_probs[index], &draft_probs[index], rng)?;
        return Ok(AcceptDecision {
            accepted: index,
            token: recovered,
        });
    }
    let bonus = sample_from_probs(&target_probs[k], rng)?;
    Ok(AcceptDecision {
        accepted: k,
        token: bonus,
    })
}

fn sample_residual(target: &[f32], draft: &[f32], rng: &mut impl Rng) -> Result<u32, Error> {
    let mut residual = vec![0.0; target.len()];
    let mut sum = 0.0_f32;
    for (index, (p, q)) in target.iter().zip(draft).enumerate() {
        let weight = (p - q).max(0.0);
        residual[index] = weight;
        sum += weight;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return sample_from_probs(target, rng);
    }
    for weight in &mut residual {
        *weight /= sum;
    }
    sample_from_probs(&residual, rng)
}

#[cfg(test)]
struct SpecRun {
    tokens: Vec<u32>,
    stats: SpeculativeStats,
    free_blocks: usize,
    rollback_blocks: usize,
}

#[cfg(test)]
fn account(blocks: &BlockSpaceManager, table: &[u32], addressed: usize) -> Result<(), Error> {
    let need = blocks.blocks_for_tokens(addressed);
    if table.len() != need || blocks.free_count() + table.len() != blocks.total_blocks() {
        return Err(Error::Inference(format!(
            "KV block accounting failed: holding {} blocks for {addressed} tokens (need {need}), free {}, total {}",
            table.len(),
            blocks.free_count(),
            blocks.total_blocks()
        )));
    }
    Ok(())
}

#[cfg(test)]
fn reserve(
    blocks: &mut BlockSpaceManager,
    table: &mut Vec<u32>,
    num_tokens: usize,
) -> Result<(), Error> {
    blocks
        .reserve(table, num_tokens)
        .map_err(|err| Error::Inference(err.to_string()))
}

/// Greedy speculative generation used by the CPU tests and the ignored GPU test.
///
/// `num_blocks` must match the device cache of a [`SpecModel`] that ignores the
/// host buffer. Both caches are freed before the function returns; a leak is
/// an error rather than a silent leftover.
#[cfg(test)]
fn speculative_generate<M: SpecModel>(
    target: &mut M,
    draft: &mut M,
    prompt: &[u32],
    max_new: usize,
    k: usize,
    block_size: usize,
    num_blocks: usize,
) -> Result<SpecRun, Error> {
    if prompt.is_empty() || max_new == 0 || k == 0 || block_size == 0 || num_blocks == 0 {
        return Err(Error::Inference(
            "speculative generation needs a prompt, a token budget, and a non-zero cache".into(),
        ));
    }
    if target.vocab_size() != draft.vocab_size() {
        return Err(Error::Config(format!(
            "draft vocab size {} is not compatible with target vocab size {}",
            draft.vocab_size(),
            target.vocab_size()
        )));
    }
    let mut target_cache = PagedKvCache::new(target.cache_geometry(num_blocks, block_size))
        .map_err(|err| Error::Inference(err.to_string()))?;
    let mut draft_cache = PagedKvCache::new(draft.cache_geometry(num_blocks, block_size))
        .map_err(|err| Error::Inference(err.to_string()))?;
    let mut target_blocks = BlockSpaceManager::new(num_blocks, block_size)
        .map_err(|err| Error::Inference(err.to_string()))?;
    let mut draft_blocks = BlockSpaceManager::new(num_blocks, block_size)
        .map_err(|err| Error::Inference(err.to_string()))?;
    let mut target_table = Vec::new();
    let mut draft_table = Vec::new();
    reserve(&mut target_blocks, &mut target_table, prompt.len())?;
    reserve(&mut draft_blocks, &mut draft_table, prompt.len())?;
    let target_prefix = target.forward_spec(&mut target_cache, prompt, 0, &target_table, false)?;
    let draft_prefix = draft.forward_spec(&mut draft_cache, prompt, 0, &draft_table, false)?;
    let mut target_len = prompt.len();
    let mut draft_len = prompt.len();
    account(&target_blocks, &target_table, prompt.len())?;
    account(&draft_blocks, &draft_table, prompt.len())?;

    let mut carry_target = Some(
        target_prefix
            .into_iter()
            .next()
            .ok_or_else(|| Error::Inference("prefill returned no logits".into()))?,
    );
    let mut carry_draft = Some(
        draft_prefix
            .into_iter()
            .next()
            .ok_or_else(|| Error::Inference("draft prefill returned no logits".into()))?,
    );
    let params = SamplingParams::greedy();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut out = Vec::new();
    let mut stats = SpeculativeStats::default();
    let mut rollback_blocks = 0_usize;

    while out.len() < max_new {
        if target_len != draft_len {
            return Err(Error::Inference(format!(
                "draft cache {draft_len} diverged from target cache {target_len}"
            )));
        }
        let remaining = max_new - out.len();
        let k_step = k.min(remaining.saturating_sub(1));
        if k_step == 0 {
            let logits = if let Some(logits) = carry_target.take() {
                let _ = carry_draft.take();
                logits
            } else {
                let pending = *out
                    .last()
                    .expect("a pending token exists after the first step");
                let logits = forward_one(
                    target,
                    &mut target_cache,
                    &mut target_blocks,
                    &mut target_table,
                    &mut target_len,
                    pending,
                )?;
                let _draft_logits = forward_one(
                    draft,
                    &mut draft_cache,
                    &mut draft_blocks,
                    &mut draft_table,
                    &mut draft_len,
                    pending,
                )?;
                logits
            };
            out.push(checked_argmax(&logits)? as u32);
            let seq_len = prompt.len() + out.len();
            reserve(&mut target_blocks, &mut target_table, seq_len)?;
            reserve(&mut draft_blocks, &mut draft_table, seq_len)?;
            account(&target_blocks, &target_table, seq_len)?;
            account(&draft_blocks, &draft_table, seq_len)?;
            continue;
        }

        let start = if let Some(logits) = carry_draft.take() {
            DraftStart::Logits(logits)
        } else {
            DraftStart::Pending(*out.last().expect("pending token"))
        };
        let proposal = draft_propose(
            draft,
            &mut draft_cache,
            &mut draft_blocks,
            &mut draft_table,
            &mut draft_len,
            start,
            k_step,
            &params,
            &mut rng,
        )?;
        let pending = if carry_target.is_some() {
            None
        } else {
            Some(*out.last().expect("pending token"))
        };
        let mut plan = plan_verify(target_len, pending, carry_target.take(), &proposal.tokens)?;
        let forward_end = plan.position + plan.tokens.len();
        let full_seq = plan.base_len + k_step + 1;
        reserve(
            &mut target_blocks,
            &mut target_table,
            forward_end.max(full_seq),
        )?;
        reserve(&mut draft_blocks, &mut draft_table, full_seq)?;
        let mut rows = target.forward_spec(
            &mut target_cache,
            &plan.tokens,
            plan.position,
            &target_table,
            true,
        )?;
        if let Some(prefix) = plan.prefix_logits.take() {
            rows.insert(0, prefix);
        }
        let decision = accept_greedy(&rows, &proposal.tokens)?;
        stats.record(k_step as u64, decision.accepted as u64);
        let keep = keep_kv_len(plan.base_len, decision.accepted);
        let seq_len = keep + 1;
        rollback_blocks += target_blocks
            .truncate(&mut target_table, seq_len)
            .map_err(|err| Error::Inference(err.to_string()))?;
        rollback_blocks += draft_blocks
            .truncate(&mut draft_table, seq_len)
            .map_err(|err| Error::Inference(err.to_string()))?;
        target_len = keep;
        draft_len = keep;
        out.extend(decision.committed(&proposal.tokens)?);
        if prompt.len() + out.len() != seq_len {
            return Err(Error::Inference(
                "speculative commit did not match the KV rollback length".into(),
            ));
        }
        account(&target_blocks, &target_table, seq_len)?;
        account(&draft_blocks, &draft_table, seq_len)?;
    }

    target_blocks
        .free_ids(&target_table)
        .map_err(|err| Error::Inference(err.to_string()))?;
    draft_blocks
        .free_ids(&draft_table)
        .map_err(|err| Error::Inference(err.to_string()))?;
    if target_blocks.free_count() != num_blocks || draft_blocks.free_count() != num_blocks {
        return Err(Error::Inference(format!(
            "KV blocks leaked after completion: target free {}, draft free {}, pool {num_blocks}",
            target_blocks.free_count(),
            draft_blocks.free_count()
        )));
    }
    Ok(SpecRun {
        tokens: out,
        stats,
        free_blocks: target_blocks.free_count(),
        rollback_blocks,
    })
}

#[cfg(test)]
fn greedy_generate<M: SpecModel>(
    model: &mut M,
    prompt: &[u32],
    max_new: usize,
    block_size: usize,
) -> Result<Vec<u32>, Error> {
    let n_blocks = (prompt.len() + max_new).div_ceil(block_size).max(1);
    let mut cache = PagedKvCache::new(model.cache_geometry(n_blocks, block_size))
        .map_err(|err| Error::Inference(err.to_string()))?;
    let table: Vec<u32> = (0..n_blocks as u32).collect();
    let mut logits = model
        .forward_spec(&mut cache, prompt, 0, &table, false)?
        .pop()
        .ok_or_else(|| Error::Inference("prefill returned no logits".into()))?;
    let mut out = Vec::new();
    while out.len() < max_new {
        let token = checked_argmax(&logits)? as u32;
        out.push(token);
        if out.len() == max_new {
            break;
        }
        let position = prompt.len() + out.len() - 1;
        logits = model
            .forward_spec(&mut cache, &[token], position, &table, false)?
            .pop()
            .ok_or_else(|| Error::Inference("decode returned no logits".into()))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::quant::QuantMode;
    use rand::RngCore;

    fn tiny(layers: usize, hidden: usize, vocab: usize) -> ModelConfig {
        ModelConfig {
            hidden_size: hidden,
            intermediate_size: hidden * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: vocab,
            tie_word_embeddings: false,
            ..ModelConfig::dummy()
        }
    }

    fn prompt() -> Vec<u32> {
        vec![1, 4, 2, 7, 3, 5]
    }

    struct Scripted {
        draws: Vec<u32>,
        at: usize,
    }

    impl RngCore for Scripted {
        fn next_u32(&mut self) -> u32 {
            let value = self.draws.get(self.at).copied().unwrap_or(0);
            self.at += 1;
            value
        }
        fn next_u64(&mut self) -> u64 {
            u64::from(self.next_u32())
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for byte in dest {
                *byte = 0;
            }
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    fn unit_bits(unit: f32) -> u32 {
        let scaled = (unit * ((1_u32 << 24) as f32)).round() as u32;
        scaled.min((1_u32 << 24) - 1) << 8
    }

    fn one_hot(vocab: usize, index: usize) -> Vec<f32> {
        let mut row = vec![0.0; vocab];
        row[index] = 1.0;
        row
    }

    #[test]
    fn greedy_accepts_the_matching_prefix_and_breaks_ties_low() {
        let rows = vec![
            vec![0.0, 2.0, 2.0],
            vec![0.0, 0.0, 5.0],
            vec![9.0, 1.0, 1.0],
        ];
        // argmax of row 0 is 1, not 2.
        let matched = accept_greedy(&rows, &[1, 2]).unwrap();
        assert_eq!(matched.accepted, 2);
        assert_eq!(matched.token, 0);
        let partial = accept_greedy(&rows, &[1, 0]).unwrap();
        assert_eq!(partial.accepted, 1);
        assert_eq!(partial.token, 2);
        let rejected = accept_greedy(&rows[..2], &[0]).unwrap();
        assert_eq!(rejected.accepted, 0);
        assert_eq!(rejected.token, 1);
        assert!(accept_greedy(&rows, &[1, 2, 0]).is_err());
        assert!(accept_greedy(&[vec![f32::NAN, 1.0], vec![0.0, 1.0]], &[0]).is_err());
    }

    #[test]
    fn rejection_sampling_accepts_a_matching_proposal_and_takes_the_bonus() {
        let vocab = 3;
        let target = vec![one_hot(vocab, 1), one_hot(vocab, 2), one_hot(vocab, 0)];
        let draft = vec![one_hot(vocab, 1), one_hot(vocab, 2)];
        let mut rng = Scripted {
            draws: vec![0, 0, 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[1, 2], &draft, &mut rng).unwrap();
        assert_eq!(decision.accepted, 2);
        assert_eq!(decision.token, 0);
        assert_eq!(decision.committed(&[1, 2]).unwrap(), vec![1, 2, 0]);
    }

    #[test]
    fn rejection_sampling_rejects_immediately_and_renormalizes_the_residual() {
        // p(token 0) / q(token 0) = 0, so the first uniform draw rejects.
        let target = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let draft = vec![vec![1.0, 0.0, 0.0]];
        let mut rng = Scripted {
            draws: vec![0, 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[0], &draft, &mut rng).unwrap();
        assert_eq!(decision.accepted, 0);
        assert_eq!(decision.token, 1);

        // ratio = 0.2/0.6. A draw of 0.5 rejects. Residual is [0, 0.4, 0].
        let target = vec![vec![0.2, 0.8, 0.0], one_hot(3, 0)];
        let draft = vec![vec![0.6, 0.4, 0.0]];
        let mut rng = Scripted {
            draws: vec![unit_bits(0.5), 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[0], &draft, &mut rng).unwrap();
        assert_eq!(decision.accepted, 0);
        assert_eq!(decision.token, 1);

        // The same ratio accepts when the draw is 0, and the bonus is token 2.
        let target = vec![vec![0.2, 0.8, 0.0], one_hot(3, 2)];
        let mut rng = Scripted {
            draws: vec![0, 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[0], &draft, &mut rng).unwrap();
        assert_eq!(decision.accepted, 1);
        assert_eq!(decision.token, 2);
    }

    #[test]
    fn rejection_sampling_covers_zero_mass_and_shape_errors() {
        // q(x) = 0 and p(x) > 0 accepts. The draw is in [0, 1).
        let target = vec![vec![0.0, 1.0], one_hot(2, 0)];
        let draft = vec![vec![0.0, 1.0]];
        let mut rng = Scripted {
            draws: vec![unit_bits(0.99), 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[1], &draft, &mut rng).unwrap();
        assert_eq!(
            decision,
            AcceptDecision {
                accepted: 1,
                token: 0
            }
        );

        // q(x) = p(x) = 0 rejects, and a zero residual falls back to p.
        let target = vec![vec![0.0, 1.0], one_hot(2, 0)];
        let draft = vec![vec![0.0, 1.0]];
        let mut rng = Scripted {
            draws: vec![0, 0],
            at: 0,
        };
        let decision = reject_sample(&target, &[0], &draft, &mut rng).unwrap();
        assert_eq!(decision.accepted, 0);
        assert_eq!(decision.token, 1);

        assert!(reject_sample(&target, &[], &[], &mut rng).is_err());
        assert!(reject_sample(&[vec![0.5, 0.5]], &[0], &[vec![0.5]], &mut rng).is_err());
        let negative = vec![vec![-0.1, 1.1], one_hot(2, 0)];
        assert!(reject_sample(&negative, &[1], &draft, &mut rng).is_err());
        assert!(plan_verify(3, Some(1), Some(vec![0.0]), &[4]).is_err());
        assert!(plan_verify(3, None, None, &[4]).is_err());
    }

    #[test]
    fn multi_token_logits_match_stepwise_decoding() {
        let cfg = tiny(2, 16, 32);
        let tokens = [1_u32, 4, 9, 2, 8, 3, 7];
        let block_size = 3;
        let n_blocks = tokens.len().div_ceil(block_size);
        let table: Vec<u32> = (0..n_blocks as u32).collect();
        let mut parallel = Model::from_random(&cfg, 11).unwrap();
        let mut cache = PagedKvCache::with_dims(
            n_blocks,
            block_size,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        )
        .unwrap();
        // Prefill a prefix, then score the rest in one chunk. That is the
        // verification forward, including a block boundary in the middle.
        parallel
            .forward_token_logits(&tokens[..3], 0, &mut cache, &table, false)
            .unwrap();
        let rows = parallel
            .forward_token_logits(&tokens[3..], 3, &mut cache, &table, true)
            .unwrap();
        assert_eq!(rows.len(), tokens.len() - 3);

        let mut stepwise = Model::from_random(&cfg, 11).unwrap();
        let mut stepped = PagedKvCache::with_dims(
            n_blocks,
            block_size,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        )
        .unwrap();
        stepwise
            .forward_token_logits(&tokens[..3], 0, &mut stepped, &table, false)
            .unwrap();
        for (offset, token) in tokens[3..].iter().enumerate() {
            let row = stepwise
                .forward_token_logits(&[*token], 3 + offset, &mut stepped, &table, false)
                .unwrap();
            assert_eq!(row[0], rows[offset], "position {}", 3 + offset);
        }
    }

    #[test]
    fn greedy_speculative_matches_the_target_token_for_token() {
        let target_cfg = tiny(2, 32, 32);
        let prompt = prompt();
        let max_new = 12;
        let k = 4;
        let blocks = 48;

        let mut reference = Model::from_random(&target_cfg, 7).unwrap();
        let expect = greedy_generate(&mut reference, &prompt, max_new, 4).unwrap();

        let mut target = Model::from_random(&target_cfg, 7).unwrap();
        let mut draft = Model::from_random(&target_cfg, 7).unwrap();
        let same =
            speculative_generate(&mut target, &mut draft, &prompt, max_new, k, 4, blocks).unwrap();
        assert_eq!(same.tokens, expect);
        assert_eq!(same.stats.accepted_tokens, same.stats.drafted_tokens);
        assert!(same.stats.steps > 0);
        assert_eq!(same.stats.acceptance_rate(), 1.0);
        assert_eq!(same.free_blocks, blocks);

        // INT8 weights change the greedy path, and speculation has to follow
        // that path rather than the f32 one.
        let mut reference = Model::from_random_with(&target_cfg, 7, QuantMode::Int8).unwrap();
        let expect_i8 = greedy_generate(&mut reference, &prompt, max_new, 4).unwrap();
        let mut target = Model::from_random_with(&target_cfg, 7, QuantMode::Int8).unwrap();
        let mut draft = Model::from_random_with(&target_cfg, 7, QuantMode::Int8).unwrap();
        let quant =
            speculative_generate(&mut target, &mut draft, &prompt, max_new, k, 4, blocks).unwrap();
        assert_eq!(quant.tokens, expect_i8);
        assert_eq!(quant.stats.acceptance_rate(), 1.0);
        assert_eq!(quant.free_blocks, blocks);

        let mut target = Model::from_random(&target_cfg, 7).unwrap();
        let draft_cfg = tiny(1, 16, 32);
        let mut draft = Model::from_random(&draft_cfg, 19).unwrap();
        let partial =
            speculative_generate(&mut target, &mut draft, &prompt, max_new, k, 1, blocks).unwrap();
        assert_eq!(partial.tokens, expect, "a weaker draft must stay lossless");
        assert!(
            partial.stats.accepted_tokens < partial.stats.drafted_tokens,
            "expected a rejection, stats {:?}",
            partial.stats
        );
        assert!(
            partial.rollback_blocks > 0,
            "rejected draft tokens must free KV blocks"
        );
        assert_eq!(partial.free_blocks, blocks);
        assert!(ensure_draft_compatible(
            &Model::from_random(&target_cfg, 1).unwrap(),
            &Model::from_random(&tiny(1, 16, 30), 1).unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("not compatible"));
    }

    #[test]
    fn kv_blocks_return_after_rollbacks_and_completion() {
        let target_cfg = tiny(2, 16, 32);
        let prompt = prompt();
        // One token per block, so every rejected draft token is its own block.
        let block_size = 1;
        let blocks = 64;
        let mut target = Model::from_random(&target_cfg, 3).unwrap();
        let mut draft = Model::from_random(&tiny(1, 16, 32), 8).unwrap();
        let run = speculative_generate(&mut target, &mut draft, &prompt, 16, 5, block_size, blocks)
            .unwrap();
        assert_eq!(run.free_blocks, blocks);
        assert!(run.rollback_blocks > 0);
        assert_eq!(run.tokens.len(), 16);

        // Same weights still have to give every block back, including the
        // blocks touched when a step crosses several pages.
        let mut target = Model::from_random(&target_cfg, 3).unwrap();
        let mut draft = Model::from_random(&target_cfg, 3).unwrap();
        let same =
            speculative_generate(&mut target, &mut draft, &prompt, 16, 5, 2, blocks).unwrap();
        assert_eq!(same.free_blocks, blocks);
        assert_eq!(same.stats.acceptance_rate(), 1.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn gpu_speculative_matches_gpu_greedy() {
        use crate::GpuDType;

        let cfg = tiny(2, 32, 32);
        let prompt = prompt();
        let max_new = 8;
        let k = 4;
        let pool = 32_usize;
        let cases = [
            (QuantMode::None, GpuDType::F32),
            (QuantMode::None, GpuDType::Bf16),
            (QuantMode::Int8, GpuDType::F32),
            (QuantMode::Int8, GpuDType::Bf16),
        ];
        for (quant, dtype) in cases {
            let mut greedy_model = Model::from_random_with(&cfg, 5, quant)
                .unwrap()
                .into_gpu_with(pool, 4, dtype)
                .unwrap();
            let expect = greedy_generate(&mut greedy_model, &prompt, max_new, 4).unwrap();
            let mut target = Model::from_random_with(&cfg, 5, quant)
                .unwrap()
                .into_gpu_with(pool, 4, dtype)
                .unwrap();
            let mut draft = Model::from_random_with(&cfg, 5, quant)
                .unwrap()
                .into_gpu_with(pool, 4, dtype)
                .unwrap();
            let same = speculative_generate(&mut target, &mut draft, &prompt, max_new, k, 4, pool)
                .unwrap();
            assert_eq!(
                same.tokens, expect,
                "full accept quant {quant:?} dtype {dtype:?}"
            );
            assert_eq!(same.stats.accepted_tokens, same.stats.drafted_tokens);
            assert_eq!(same.free_blocks, pool);

            let mut target = Model::from_random_with(&cfg, 5, quant)
                .unwrap()
                .into_gpu_with(pool, 2, dtype)
                .unwrap();
            let mut draft = Model::from_random_with(&tiny(1, 16, 32), 9, quant)
                .unwrap()
                .into_gpu_with(pool, 2, dtype)
                .unwrap();
            let partial =
                speculative_generate(&mut target, &mut draft, &prompt, max_new, k, 2, pool)
                    .unwrap();
            assert_eq!(
                partial.tokens, expect,
                "rollback quant {quant:?} dtype {dtype:?}"
            );
            assert!(
                partial.rollback_blocks > 0,
                "quant {quant:?} dtype {dtype:?} did not free a rolled-back block"
            );
            assert_eq!(partial.free_blocks, pool);
        }
    }
}

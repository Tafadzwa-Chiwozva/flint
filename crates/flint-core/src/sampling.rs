//! Greedy, temperature, top-k, and top-p sampling.
//!
//! Processors run in the same order as Hugging Face's default warpers:
//! temperature, then top-k, then top-p, then a categorical draw. Temperature
//! `0` is greedy and ignores top-k and top-p. Ties in greedy argmax keep the
//! lowest index.

use rand::Rng;
use rand::SeedableRng;

use crate::error::Error;

/// How to turn a logit vector into the next token.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// `0` selects the argmax. Otherwise logits are divided by this value.
    pub temperature: f32,
    /// Keep only the `k` largest logits. `None` disables the filter.
    pub top_k: Option<usize>,
    /// Nucleus probability in `(0, 1]`. `None` or `1` disables the filter.
    pub top_p: Option<f32>,
    /// RNG seed. `None` draws entropy from the OS.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
        }
    }

    /// Reject temperatures, top-k, and top-p values that have no meaning.
    pub fn validate(&self) -> Result<(), Error> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::Sampling(
                "temperature must be finite and >= 0".into(),
            ));
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(Error::Sampling("top_k must be >= 1 when set".into()));
            }
        }
        if let Some(p) = self.top_p {
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err(Error::Sampling("top_p must be in (0, 1]".into()));
            }
        }
        Ok(())
    }

    /// Build the RNG this request will sample from.
    #[must_use]
    pub fn rng(&self) -> rand::rngs::StdRng {
        match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        }
    }
}

/// Sample one token.
///
/// Non-finite logits are an error: a NaN in the forward pass should fail the
/// request instead of silently emitting token 0.
pub fn sample(logits: &[f32], params: &SamplingParams, rng: &mut impl Rng) -> Result<u32, Error> {
    if logits.is_empty() {
        return Err(Error::Sampling("empty logits".into()));
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(Error::Sampling("non-finite logit".into()));
    }
    params.validate()?;
    if params.temperature == 0.0 {
        return Ok(argmax(logits) as u32);
    }

    let inv_temp = 1.0 / params.temperature;
    let mut scored: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, logit)| (index, logit * inv_temp))
        .collect();

    if let Some(k) = params.top_k {
        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(k);
        }
    }

    if let Some(top_p) = params.top_p {
        if top_p < 1.0 {
            scored = nucleus(scored, top_p);
        }
    }

    Ok(categorical(&scored, rng)? as u32)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > best {
            best = logit;
            best_index = index;
        }
    }
    best_index
}

fn nucleus(mut scored: Vec<(usize, f32)>, top_p: f32) -> Vec<(usize, f32)> {
    let max = scored
        .iter()
        .map(|item| item.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut order: Vec<(usize, f32, f32)> = scored
        .iter()
        .map(|&(index, logit)| (index, logit, (logit - max).exp()))
        .collect();
    let sum: f32 = order.iter().map(|item| item.2).sum();
    if sum == 0.0 || !sum.is_finite() {
        scored.truncate(1);
        return scored;
    }
    order.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut cumulative = 0.0;
    let mut kept = Vec::new();
    for (index, logit, prob) in order {
        cumulative += prob / sum;
        kept.push((index, logit));
        if cumulative >= top_p {
            break;
        }
    }
    kept
}

fn categorical(scored: &[(usize, f32)], rng: &mut impl Rng) -> Result<usize, Error> {
    if scored.is_empty() {
        return Err(Error::Sampling(
            "sampling filter removed every token".into(),
        ));
    }
    let max = scored
        .iter()
        .map(|item| item.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(scored.len());
    let mut sum = 0.0_f32;
    for item in scored {
        let weight = (item.1 - max).exp();
        weights.push(weight);
        sum += weight;
    }
    if sum == 0.0 || !sum.is_finite() {
        return Ok(scored[0].0);
    }
    let mut threshold = rng.gen::<f32>() * sum;
    for (item, weight) in scored.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(item.0);
        }
    }
    Ok(scored[scored.len() - 1].0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn greedy_picks_argmax_and_breaks_ties_toward_zero() {
        assert_eq!(
            sample(
                &[1.0, 3.0, 3.0, 2.0],
                &SamplingParams::greedy(),
                &mut rng(1)
            )
            .unwrap(),
            1
        );
        let mut params = SamplingParams::greedy();
        params.top_k = Some(1);
        params.top_p = Some(0.1);
        assert_eq!(sample(&[0.0, 5.0, 1.0], &params, &mut rng(1)).unwrap(), 1);
    }

    #[test]
    fn temperature_zero_matches_greedy() {
        let logits = [0.2, -1.0, 0.4, 0.1];
        let zero = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        assert_eq!(
            sample(&logits, &zero, &mut rng(2)).unwrap(),
            sample(&logits, &SamplingParams::greedy(), &mut rng(3)).unwrap()
        );
    }

    #[test]
    fn top_k_never_returns_outside_the_kept_set() {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(2),
            top_p: None,
            seed: Some(0),
        };
        for seed in 0..40 {
            let token = sample(&logits, &params, &mut rng(seed)).unwrap();
            assert!(token == 3 || token == 4, "token {token}");
        }
    }

    #[test]
    fn top_p_drops_the_tail() {
        // softmax of these is approximately [0.5, 0.3, 0.2] after normalization
        // of exp(logits) if we set logits = log(weights).
        let logits = [0.5_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln()];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.6),
            seed: None,
        };
        for seed in 0..30 {
            let token = sample(&logits, &params, &mut rng(seed)).unwrap();
            assert!(
                token == 0 || token == 1,
                "token {token} escaped the nucleus"
            );
        }
    }

    #[test]
    fn same_seed_is_stable_and_different_seeds_can_differ() {
        let logits = [0.0, 0.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: Some(7),
        };
        let first = sample(&logits, &params, &mut rng(7)).unwrap();
        let second = sample(&logits, &params, &mut rng(7)).unwrap();
        assert_eq!(first, second);
        let mut seen = std::collections::HashSet::new();
        for seed in 0..24 {
            seen.insert(sample(&logits, &params, &mut rng(seed)).unwrap());
        }
        assert!(seen.len() > 1, "expected more than one token, got {seen:?}");
    }

    #[test]
    fn rejects_bad_parameters_and_nan_logits() {
        let mut params = SamplingParams {
            temperature: -0.1,
            ..SamplingParams::default()
        };
        assert!(params.validate().is_err());
        params.temperature = 1.0;
        params.top_p = Some(0.0);
        assert!(params.validate().is_err());
        params.top_p = None;
        assert!(sample(&[0.0, f32::NAN], &params, &mut rng(1)).is_err());
    }
}

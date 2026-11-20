/// Errors from configuration, weights, tokenization, sampling, or the forward pass.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `config.json` was missing a field or described an unsupported variant.
    #[error("config: {0}")]
    Config(String),
    /// A safetensors tensor was missing, misshapen, or an unsupported dtype.
    #[error("weights: {0}")]
    Weights(String),
    /// The Hugging Face tokenizer could not encode or decode.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The forward pass was given an inconsistent cache or token id.
    #[error("inference: {0}")]
    Inference(String),
    /// Sampling parameters or logits were unusable.
    #[error("sampling: {0}")]
    Sampling(String),
}

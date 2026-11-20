//! Hugging Face tokenizer plus the Qwen2.5-Instruct chat template.
//!
//! The template is implemented directly rather than through Jinja. It covers
//! the no-tools path of `Qwen2.5-0.5B-Instruct`'s `chat_template`: a default
//! system prompt when the conversation does not open with one, then each
//! message, then the assistant generation tag. Tool-call markup is rejected.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::Error;

/// One chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Plain text. Multimodal parts are flattened by the HTTP layer.
    pub content: String,
}

/// Render messages the way Qwen2.5-Instruct does when no tools are supplied.
pub fn apply_qwen_chat_template(messages: &[ChatTurn]) -> Result<String, Error> {
    if messages.is_empty() {
        return Err(Error::Tokenizer(
            "chat template needs at least one message".into(),
        ));
    }
    let mut out = String::new();
    if messages[0].role == "system" {
        out.push_str("<|im_start|>system\n");
        out.push_str(&messages[0].content);
        out.push_str("<|im_end|>\n");
    } else {
        out.push_str(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n",
        );
    }
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && message.role == "system" {
            continue;
        }
        if message.role != "user" && message.role != "assistant" && message.role != "system" {
            return Err(Error::Tokenizer(format!(
                "unsupported chat role '{}'; phase 1 accepts system, user, and assistant",
                message.role
            )));
        }
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

/// A `tokenizer.json` plus the eos ids declared next to it.
pub struct FlintTokenizer {
    inner: Tokenizer,
    eos_ids: Vec<u32>,
}

impl FlintTokenizer {
    /// Load `tokenizer.json` from a model directory.
    ///
    /// `add_special_tokens` is left false at encode time. Qwen does not prepend
    /// a BOS id, and the chat template already contains `<|im_start|>` /
    /// `<|im_end|>` as literal special tokens.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|err| Error::Tokenizer(format!("load {}: {err}", path.display())))?;
        let mut eos_ids = read_eos_ids(&dir.join("config.json"));
        for id in read_eos_ids(&dir.join("generation_config.json")) {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        Ok(Self { inner, eos_ids })
    }

    /// Encode text. Special tokens that appear in the string stay atomic.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| Error::Tokenizer(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.inner
            .decode(ids, true)
            .map_err(|err| Error::Tokenizer(err.to_string()))
    }

    /// Chat-template then encode.
    pub fn encode_chat(&self, messages: &[ChatTurn]) -> Result<Vec<u32>, Error> {
        let text = apply_qwen_chat_template(messages)?;
        self.encode(&text)
    }

    /// Eos ids found in `config.json` and `generation_config.json`.
    #[must_use]
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
}

fn read_eos_ids(path: &Path) -> Vec<u32> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    match value.get("eos_token_id") {
        Some(serde_json::Value::Number(number)) => number
            .as_u64()
            .map(|id| vec![id as u32])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_u64().map(|id| id as u32))
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_system_prompt_is_inserted_for_a_user_message() {
        let text = apply_qwen_chat_template(&[ChatTurn {
            role: "user".into(),
            content: "Hello".into(),
        }])
        .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn explicit_system_message_is_not_duplicated() {
        let text = apply_qwen_chat_template(&[
            ChatTurn {
                role: "system".into(),
                content: "Be brief.".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "Hi".into(),
            },
        ])
        .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn unknown_role_is_rejected() {
        let err = apply_qwen_chat_template(&[ChatTurn {
            role: "tool".into(),
            content: "{}".into(),
        }])
        .unwrap_err();
        assert!(err.to_string().contains("unsupported chat role"));
    }

    #[test]
    #[ignore = "requires FLINT_MODEL_DIR with tokenizer.json"]
    fn qwen_special_tokens_are_atomic() {
        let dir = std::env::var("FLINT_MODEL_DIR").expect("FLINT_MODEL_DIR");
        let tok = FlintTokenizer::open(std::path::Path::new(&dir)).unwrap();
        let start = tok.encode("<|im_start|>").unwrap();
        let end = tok.encode("<|im_end|>").unwrap();
        assert_eq!(start, vec![151644], "im_start ids {start:?}");
        assert_eq!(end, vec![151645], "im_end ids {end:?}");
        let hello = tok.encode("Hello").unwrap();
        assert!(!hello.is_empty());
    }
}

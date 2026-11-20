//! Safetensors loader.
//!
//! Checkpoints are stored as bf16 in the Qwen release. They are upcast to f32
//! once, at load time, so the reference forward pass has a single numeric
//! type. Sharded `model.safetensors.index.json` layouts are accepted; each
//! shard is dropped before the next one is read.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::error::Error;

pub struct TensorMap {
    tensors: HashMap<String, Vec<f32>>,
}

impl TensorMap {
    pub fn load_dir(dir: &Path, skip: &[&str]) -> Result<Self, Error> {
        let files = weight_files(dir)?;
        let mut tensors = HashMap::new();
        for file in files {
            load_file(&file, skip, &mut tensors)?;
        }
        Ok(Self { tensors })
    }

    pub fn take(&mut self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let values = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor {name}")))?;
        if values.len() != len {
            return Err(Error::Weights(format!(
                "{name}: expected {len} elements, found {}",
                values.len()
            )));
        }
        Ok(values)
    }

    pub fn take_optional(&mut self, name: &str, len: usize) -> Result<Option<Vec<f32>>, Error> {
        if !self.tensors.contains_key(name) {
            return Ok(None);
        }
        Ok(Some(self.take(name, len)?))
    }
}

fn weight_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Error> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let text = fs::read_to_string(&index)
            .map_err(|err| Error::Weights(format!("read {}: {err}", index.display())))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|err| Error::Weights(format!("parse {}: {err}", index.display())))?;
        let map = value
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Weights(format!("{} has no weight_map", index.display())))?;
        let mut names: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        if names.is_empty() {
            return Err(Error::Weights("weight map is empty".into()));
        }
        return Ok(names.into_iter().map(|name| dir.join(name)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(Error::Weights(format!(
        "no model.safetensors in {}",
        dir.display()
    )))
}

fn load_file(
    path: &Path,
    skip: &[&str],
    into: &mut HashMap<String, Vec<f32>>,
) -> Result<(), Error> {
    let bytes =
        fs::read(path).map_err(|err| Error::Weights(format!("read {}: {err}", path.display())))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|err| Error::Weights(format!("{}: {err}", path.display())))?;
    let names = tensors.names();
    for name in &names {
        if skip.iter().any(|skipped| skipped == name) {
            continue;
        }
        let view = tensors
            .tensor(name)
            .map_err(|err| Error::Weights(format!("{name}: {err}")))?;
        let values = view_to_f32(&view, name)?;
        if into.insert((*name).to_string(), values).is_some() {
            return Err(Error::Weights(format!("duplicate tensor {name}")));
        }
    }
    tracing::info!(path = %path.display(), tensors = names.len(), "loaded shard");
    Ok(())
}

fn view_to_f32(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>, Error> {
    let len: usize = view.shape().iter().product();
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => {
            if data.len() != len * 4 {
                return Err(Error::Weights(format!("{name}: truncated f32 payload")));
            }
            let mut out = Vec::with_capacity(len);
            for chunk in data.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if data.len() != len * 2 {
                return Err(Error::Weights(format!("{name}: truncated bf16 payload")));
            }
            let mut out = Vec::with_capacity(len);
            for chunk in data.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*chunk);
                out.push(half::bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::F16 => {
            if data.len() != len * 2 {
                return Err(Error::Weights(format!("{name}: truncated f16 payload")));
            }
            let mut out = Vec::with_capacity(len);
            for chunk in data.as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(*chunk);
                out.push(half::f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => Err(Error::Weights(format!(
            "{name}: unsupported dtype {other:?}"
        ))),
    }
}

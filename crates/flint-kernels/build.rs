//! Builds CUDA PTX when the `cuda` feature is enabled.
//!
//! With the feature off this script is a no-op, so a machine without the CUDA
//! toolkit can build, test, and serve the CPU engine. With the feature on,
//! `nvcc -ptx` compiles `kernels/ops.cu`. The PTX is compiled into the crate
//! and loaded with the driver API at runtime, so this step does not link
//! `libcudart` and does not need a GPU.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/ops.cu");
    println!("cargo:rerun-if-changed=kernels/bf16_ext.cu");
    println!("cargo:rerun-if-changed=kernels/int8.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    if let Err(err) = compile_ptx() {
        panic!("flint-kernels cuda feature is enabled but the CUDA build failed: {err}");
    }
}

fn compile_ptx() -> Result<(), String> {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let out_dir = std::env::var("OUT_DIR").map_err(|err| err.to_string())?;
    let src = "kernels/ops.cu";
    if !Path::new(src).exists() {
        return Err(format!("missing {src}"));
    }

    let version = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map_err(|err| {
            format!(
                "could not run `{nvcc}` ({err}). Install the CUDA toolkit (nvcc) or build without `--features cuda`."
            )
        })?;
    if !version.status.success() {
        return Err(format!(
            "`{nvcc} --version` failed:\n{}",
            String::from_utf8_lossy(&version.stderr)
        ));
    }

    let ptx_path = format!("{out_dir}/flint.ptx");
    // compute_75 PTX JIT-compiles on Turing and everything newer (4090, A100, H100).
    let output = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "-std=c++17",
            "-arch=compute_75",
            src,
            "-o",
            &ptx_path,
        ])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "nvcc failed while compiling {src}:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let written = std::fs::read(&ptx_path).map_err(|err| err.to_string())?;
    if written.is_empty() {
        return Err(format!("nvcc wrote an empty {ptx_path}"));
    }
    Ok(())
}

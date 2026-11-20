//! Builds the CUDA translation units when the `cuda` feature is enabled.
//!
//! With the feature off this script is a no-op, so a machine without the CUDA
//! toolkit can build, test, and serve the CPU engine.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/paged_attention.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    if let Err(err) = compile_cuda() {
        panic!("flint-kernels cuda feature is enabled but the CUDA build failed: {err}");
    }
}

fn compile_cuda() -> Result<(), String> {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let out_dir = std::env::var("OUT_DIR").map_err(|err| err.to_string())?;
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());

    let version = std::process::Command::new(&nvcc)
        .arg("--version")
        .status()
        .map_err(|err| {
            format!("could not run `{nvcc}` ({err}). Install the CUDA toolkit or unset the cuda feature.")
        })?;
    if !version.success() {
        return Err(format!("`{nvcc} --version` failed"));
    }

    let sources = ["kernels/rms_norm.cu", "kernels/paged_attention.cu"];
    let mut objects = Vec::with_capacity(sources.len());
    for src in sources {
        let stem = Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad source path {src}"))?;
        let obj = format!("{out_dir}/{stem}.o");
        let status = std::process::Command::new(&nvcc)
            .args([
                "-O3",
                "-std=c++17",
                "--compiler-options",
                "-fPIC",
                "-c",
                src,
                "-o",
                &obj,
            ])
            .status()
            .map_err(|err| err.to_string())?;
        if !status.success() {
            return Err(format!("nvcc failed while compiling {src}"));
        }
        objects.push(obj);
    }

    let lib = format!("{out_dir}/libflint_kernels.a");
    let mut ar = std::process::Command::new("ar");
    ar.arg("rcs").arg(&lib);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status().map_err(|err| format!("ar failed: {err}"))?;
    if !status.success() {
        return Err("ar failed while packing libflint_kernels.a".into());
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=flint_kernels");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}

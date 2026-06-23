use std::env;
use std::path::PathBuf;

fn main() {
    // ── 搜索 ModelDeploySDK ──
    if let Ok(dir) = env::var("MODELDEPLOY_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
        println!("cargo:rustc-link-lib=ModelDeploySDK");
    } else {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let project_root = manifest_dir.parent().unwrap().parent().unwrap();
        for rel in &["build/lib", "build/bin", "build_debug/lib", "build_debug/bin"] {
            let path = project_root.join(rel);
            if path.exists() {
                println!("cargo:rustc-link-search=native={}", path.canonicalize().unwrap().display());
            }
        }
        println!("cargo:rustc-link-lib=ModelDeploySDK");
    }

    // ── bindgen 自动生成 FFI 绑定（可选） ──
    // 设置环境变量 BINDGEN_REGEN=1 触发重新生成
    // 需要安装: cargo install bindgen-cli
    if env::var("BINDGEN_REGEN").is_ok() {
        println!("cargo:rerun-if-changed=../../capi");
        let out_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
        let status = std::process::Command::new("bindgen")
            .args(&[
                "../../capi/common/md_types.h",
                "-o", out_dir.join("ffi_autogen.rs").to_str().unwrap(),
                "--no-layout-tests",
                "--allowlist-function", "md_.*",
                "--allowlist-var", "MD_.*",
                "--allowlist-type", "MD.*",
                "--", "-x", "c", "-std=c99",
                "-I../../",
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=bindgen 生成成功: src/ffi_autogen.rs");
            }
            _ => {
                println!("cargo:warning=bindgen 不可用，使用 src/ffi.rs（手动维护）");
            }
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}

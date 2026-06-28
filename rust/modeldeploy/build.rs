use std::env;
use std::path::PathBuf;
use std::fs;

fn find_sdk_root() -> Option<PathBuf> {
    if let Ok(dir) = env::var("MODELDEPLOY_LIB_DIR") {
        // 用户指定了路径，从该目录找 SDK
        let p = PathBuf::from(&dir);
        if p.join("ModelDeploySDK.lib").exists() || p.join("ModelDeploySDK.dll").exists() {
            return Some(p.parent()?.to_path_buf());
        }
        return Some(p.parent()?.to_path_buf());
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap().parent().unwrap();
    for rel in &["build", "build_debug"] {
        let dir = project_root.join(rel);
        if dir.join("bin").join("ModelDeploySDK.dll").exists() || dir.join("lib").join("ModelDeploySDK.lib").exists() {
            return Some(dir);
        }
    }
    None
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // ── 1. 链接 ModelDeploySDK ──
    if let Some(sdk_root) = find_sdk_root() {
        for lib_dir in &[sdk_root.join("bin"), sdk_root.join("lib")] {
            if lib_dir.exists() {
                println!("cargo:rustc-link-search=native={}", lib_dir.canonicalize().unwrap().display());
            }
        }
        println!("cargo:rustc-link-lib=ModelDeploySDK");

        // ── 2. 拷贝运行时 DLL 到输出目录 ──
        // 避免系统 PATH 里的旧版 onnxruntime 干扰
        let bin_dir = sdk_root.join("bin");
        if bin_dir.exists() {
            let out_dir = PathBuf::from(env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
                manifest_dir.join("target").to_string_lossy().to_string()
            }));
            let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
            let copy_dirs = vec![
                out_dir.join(&profile),
                out_dir.join(&profile).join("examples"),
            ];

            if let Ok(entries) = fs::read_dir(&bin_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "dll" {
                            for dest_dir in &copy_dirs {
                                if dest_dir.exists() {
                                    let dest = dest_dir.join(path.file_name().unwrap());
                                    if !dest.exists() {
                                        let _ = fs::copy(&path, &dest);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        println!("cargo:warning=ModelDeploySDK not found. Build it first: cmake --build ../../build");
    }

    println!("cargo:rerun-if-changed=build.rs");
}

use std::env;
use std::path::PathBuf;

fn main() {
    // 搜索 ModelDeploySDK 库文件
    // 优先从 MODELDEPLOY_LIB_DIR 环境变量读取
    if let Ok(dir) = env::var("MODELDEPLOY_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
        println!("cargo:rustc-link-lib=ModelDeploySDK");
        return;
    }

    // 从 cargo manifest 目录向上搜索 build/ 目录
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.parent().unwrap().parent().unwrap(); // rust/modeldeploy -> rust -> ModelDeploy

    let search_paths = [
        project_root.join("build").join("lib"),
        project_root.join("build").join("bin"),
        project_root.join("build_debug").join("lib"),
        project_root.join("build_debug").join("bin"),
    ];

    let mut found = false;
    for path in &search_paths {
        if path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                path.canonicalize().unwrap().display()
            );
            found = true;
        }
    }

    if found {
        println!("cargo:rustc-link-lib=ModelDeploySDK");
    } else {
        println!("cargo:warning=ModelDeploySDK library not found. Build it first: cmake --build ../../build");
    }

    println!("cargo:rerun-if-changed=build.rs");
}

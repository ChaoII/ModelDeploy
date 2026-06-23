use std::env;
use std::path::PathBuf;

fn main() {
    // 搜索 ModelDeploySDK 库文件
    if let Ok(dir) = env::var("MODELDEPLOY_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    } else {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let project_root = manifest_dir.parent().unwrap().parent().unwrap();
        for rel in &["build/lib", "build/bin", "build_debug/lib", "build_debug/bin"] {
            let path = project_root.join(rel);
            if path.exists() {
                println!("cargo:rustc-link-search=native={}", path.canonicalize().unwrap().display());
            }
        }
    }
    println!("cargo:rustc-link-lib=ModelDeploySDK");
    println!("cargo:rerun-if-changed=build.rs");
}

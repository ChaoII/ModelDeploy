use std::env;
use std::path::PathBuf;

fn main() {
    // 搜索 ModelDeploySDK 库文件
    // 优先从 MODELDEPLOY_LIB_DIR 环境变量读取
    if let Ok(dir) = env::var("MODELDEPLOY_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    }

    // 搜索 build/ 和 build/bin/ 目录
    let search_paths = [
        "build/bin",
        "build/lib",
        "build_debug/bin",
        "build_debug/lib",
    ];

    for rel in &search_paths {
        let path = PathBuf::from(rel);
        if path.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                path.canonicalize().unwrap().display()
            );
        }
    }

    // 指定链接的库名
    println!("cargo:rustc-link-lib=dylib=ModelDeploySDK");

    // 如果需要额外依赖（如 onnxruntime、opencv），也在此链接
    // 这取决于 ModelDeploySDK 的构建配置
    // println!("cargo:rustc-link-lib=onnxruntime");
    // println!("cargo:rustc-link-lib=opencv_world490");

    println!("cargo:rerun-if-changed=build.rs");
}

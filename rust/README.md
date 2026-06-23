# modeldeploy

Rust 语言绑定，通过 CAPI 封装 [ModelDeploy](https://github.com/ChaoII/ModelDeploy) C++ 推理 SDK。

---

## 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [模块清单](#模块清单)
- [API 概览](#api-概览)
- [测试](#测试)
- [构建配置](#构建配置)
- [常见问题](#常见问题)

---

## 系统要求

| 项目 | 最低要求 |
|------|---------|
| Rust | 1.70+（edition 2021） |
| C++ 编译器 | MSVC 2022 / GCC 11+ （编译 SDK 时需要） |
| CMake | 3.20+（编译 SDK 时需要） |
| CUDA | 12.x（使用 GPU 推理时需要） |
| 操作系统 | Windows x64 / Linux x64 |

---

## 快速开始

### 1. 编译 SDK

```bash
# 从项目根目录执行
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF \
  -DWITH_GPU=OFF -DENABLE_ORT=ON -DENABLE_MNN=OFF
cmake --build build
```

编译完成后，`build/bin/` 目录下会生成 `ModelDeploySDK.dll` 以及依赖的 `onnxruntime.dll` 等。

### 2. 构建 Rust crate

```bash
cd rust/modeldeploy
cargo build --release
```

> `build.rs` 会自动搜索 `../../build/bin/` 目录下的 SDK 库文件和 DLL，并自动拷贝运行时 DLL 到目标目录。

### 3. 运行检测示例

```bash
cargo run --release --example detection
```

第一次运行会自动下载依赖并编译。检测示例默认使用 `../../test_data/test_models/yolo11n_nms.onnx` 模型和 `../../test_data/test_images/test_detection0.jpg` 图片，可直接运行。

---

## 使用方式

### 作为依赖引入

在 `Cargo.toml` 中添加：

```toml
[dependencies]
modeldeploy = { path = "path/to/modeldeploy/rust/modeldeploy" }
```

### 最小检测示例

```rust
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::detection::UltralyticsDet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置运行时
    let opt = RuntimeOption::new()
        .gpu(0)                  // GPU 设备 0
        .fp16(true)              // FP16 推理
        .enable_trt(true)        // TensorRT EP
        .ort_backend();          // ONNX Runtime 后端

    // CPU 模式:
    // let opt = RuntimeOption::new().cpu(4).ort_backend();

    // 2. 加载模型
    let model = UltralyticsDet::new("yolo11n.onnx", &opt)?;

    // 3. 读取图像
    let img = Image::read("test.jpg")?;

    // 4. 推理
    let results = model.predict(&img)?;

    // 5. 输出结果
    for r in &results {
        println!("label={} score={:.4} rect=[{}x{} {}x{}]",
                 r.label_id, r.score,
                 r.rect.x, r.rect.y, r.rect.width, r.rect.height);
    }
    Ok(())
}
```

---

## 模块清单

| 模块 | Rust 类型 | 对应 C++ 模型 | 示例 |
|------|----------|-------------|------|
| **检测** | `UltralyticsDet` | YOLO 检测 | `cargo run --example detection` |
| **分类** | `UltralyticsCls` | YOLO 分类 | `cargo run --example classification` |
| **人脸检测** | `Scrfd` | SCRFD | `cargo run --example face_detection` |
| **OBB** | `UltralyticsObb` | YOLO OBB | `cargo run --example obb` |
| **分割** | `UltralyticsSeg` | YOLO 实例分割 | `cargo run --example seg` |
| **姿态** | `UltralyticsPose` | YOLO-Pose | `cargo run --example pose` |
| **OCR** | `PaddleOcr` | PaddleOCR | — |
| **车牌识别** | `LprPipeline` | LPR Pipeline | — |
| **行人属性** | `PedestrianAttribute` | PP-Human | — |
| **人脸识别** | `FaceRec` | ArcFace | `cargo run --example face_rec` |
| **人脸年龄** | `FaceAge` | AgeNet | `cargo run --example face_age` |
| **人脸性别** | `FaceGender` | GenderNet | `cargo run --example face_gender` |
| **人脸防伪** | `FaceAntiSpoofPipeline` | AntiSpoofing | — |
| **TTS** | `Kokoro` | Kokoro TTS | — |

---

## API 概览

### RuntimeOption

```rust
let opt = RuntimeOption::new()               // 调用 CAPI md_create_default_runtime_option()
    .gpu(device_id)                           // 使用 GPU（默认 CPU）
    .cpu(threads)                             // 使用 CPU + 设置线程数
    .fp16(true/false)                         // 启用/禁用 FP16
    .enable_trt(true/false)                   // 启用/禁用 TensorRT EP
    .ort_backend()                            // ONNX Runtime 后端
    .mnn_backend()                            // MNN 后端
    .trt_backend()                            // TensorRT 后端
    .trt_cache("./path")                      // TRT engine 缓存路径
    .trt_min_shape("images:1x3x640x640")      // TRT 动态 shape（最小）
    .password("123456")                       // 加密模型密码
    .ort_log_level(3);                        // ORT 日志级别 (0=verbose..4=fatal)
```

### 图像操作

```rust
let img = Image::read("path.jpg")?;               // 从文件读取
let img = Image::from_bgr24(&data, w, h);          // 从 BGR 数据创建（零拷贝）
let img = Image::from_nv12(&nv12_buf, w, h)?;      // 从 NV12 转换

img.width();                                       // 图像宽度
img.height();                                      // 图像高度
img.channels();                                    // 通道数
img.data();                                        // 像素数据切片
img.save("out.jpg")?;                              // 保存到文件
img.clone_image()?;                                // 深拷贝
```

### 模型推理模式

所有模型都遵循相同的生命周期：

```rust
// 1. 创建（RAII，Drop 自动释放）
let model = UltralyticsDet::new("model.onnx", &opt)?;

// 2. 推理
let results = model.predict(&image)?;

// 3. 使用结果（for 循环等）
for r in &results {
    println!("{} {:.4}", r.label_id, r.score);
}

// 4. 自动释放：model 超出作用域时 Drop 调用 md_free_*_model()
```

---

## 测试

### 运行全部测试

```bash
cargo test
```

### 测试覆盖

| 测试类型 | 数量 | 说明 |
|---------|------|------|
| 单元测试 | 8 | Rust 类型系统、错误码转换、RuntimeOption |
| CAPI 签名测试 | 70 | 验证每个 CAPI 函数的链接和参数类型正确 |

包含 CAPI 签名测试需要 SDK DLL 在 `build/bin/` 中。

### CI

```yaml
# .github/workflows/rust.yml
# 在 main 分支上 rust/** 或 capi/** 变更时自动触发
# 1. 编译 SDK（WITH_GPU=OFF，仅 ORT 后端）
# 2. cargo build + clippy
# 3. cargo test
```

---

## 构建配置

### 环境变量

| 变量 | 作用 | 示例 |
|------|------|------|
| `MODELDEPLOY_LIB_DIR` | 指定 SDK 库路径 | `C:/libs` |
| `BINDGEN_REGEN` | 重新生成 FFI 绑定 | `1` |

### 手动指定 SDK 路径

```bash
export MODELDEPLOY_LIB_DIR=/path/to/build/bin
cargo build
```

---

## 常见问题

### 1. `TensorrtExecutionProvider enabled` 关不掉

`enable_trt(false)` 无效？检查 `target/release/` 下是否有旧版 `ModelDeploySDK.dll`。

```
# 清理旧 DLL 后重新编译
rm -rf target/release/*.dll
cargo build --release
```

### 2. ORT 版本冲突

```text
The requested API version [22] is not available, only API versions [1, 17] ...
```

系统 PATH 中有旧版 `onnxruntime.dll`。`build.rs` 会自动拷贝正确版本到目标目录，如果还出现请检查：

```bash
# 确认 target/release/ 下有 onnxruntime.dll
ls target/release/onnxruntime.dll
```

### 3. `STATUS_ACCESS_VIOLATION` 崩溃

- SDK DLL 与运行时加载的 DLL 版本不匹配
- 清理 `target/` 后重新编译
- 确保 `build/bin/` 下的 DLL 是最新编译的

### 4. `MDStatusCode` 常量命名警告

```text
warning: constant `MDStatusCode_Success` should have an upper case name
```

CAPI 常量命名风格与 Rust 惯例不同，已通过 `#[allow(non_upper_case_globals)]` 抑制，不影响使用。

### 5. 添加新模型

当 CAPI 新增函数时，需要在 `src/ffi.rs` 中手动添加对应的 `extern "C"` 声明：

```rust
extern "C" {
    pub fn md_create_new_model(
        model: *mut MDModel,
        path: *const c_char,
        option: *const MDRuntimeOption,
    ) -> MDStatusCode;
}
```

然后在 `src/vision/` 或 `src/audio/` 下添加对应的安全封装模块，遵循现有模块的 RAII 模式。

### 6. 跨平台

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows x64 | ✅ 已验证 | MSVC + Ninja |
| Linux x64 | ✅ CI 已验证 | GCC + Ninja |
| macOS | ⏳ 待验证 | 需自行编译 SDK |

---

## 许可证

MIT License。与 ModelDeploy 项目一致。

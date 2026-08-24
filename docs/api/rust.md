# Rust 绑定

Rust 通过 FFI 封装 C API。目录 `rust/modeldeploy/`：

```rust
use modeldeploy::runtime::RuntimeOption;

let mut option = RuntimeOption::new();
option.ort_backend();   // 或 sophgo_backend(0)/trt_backend()/mnn_backend()
```

## 主要模块文件

| 文件 | 说明 |
|------|------|
| `src/runtime.rs` | `RuntimeOption` 封装 |
| `src/ffi.rs` | FFI 声明（`MD_BACKEND_*` 常量等） |
| `src/model.rs` | 模型加载/推理 |
| `src/audio.rs` | 音频能力封装 |
| `src/barcode.rs` | 条码/二维码封装 |
| `src/nlp.rs` | NLP 能力封装 |
| `src/solution.rs` | 解决方案封装 |
| `src/tracker.rs` | 跟踪能力封装 |
| `src/image.rs` | 图像处理 |
| `src/types.rs` / `src/error.rs` | 类型与错误 |

## 示例（运行于 `rust/modeldeploy/examples/`）

内置 11 个示例：`classification` / `depth` / `detection` / `face_age` / `face_detection` / `face_gender` / `face_rec` / `obb` / `pose` / `seg` / `sem`。

```bash
cd rust/modeldeploy
cargo run --example detection
```

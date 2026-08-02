# Vision Processor Backend 抽象 + Sophgo(算能) 推理后端集成

## 背景与目标

ModelDeploy 当前的前后处理是每个模型 preprocessor 内部硬编码 `yolo_preprocess_cpu/cuda`，靠 `use_cuda_preproc_` 布尔开关二选一。推理后端（ORT/MNN/TRT）是独立的 `BaseBackend` 抽象，但前后处理没有对应的抽象。

目标：
1. 建立 **前后处理算子抽象** `VisionProcessorBackend`，让 CPU / CPU-SIMD / CUDA / Sophgo-TPU / 未来华为 NPU、瑞芯微 RKNPU 都能插拔实现。
2. 集成 **Sophgo 算能后端**（目标硬件：SE9 微服务器，BM1688 或 CV186AH）：
   - 推理：SOPHON-Sail `sail::Engine` 加载 .bmodel 跑推理
   - 预处理：BMCV（resize/letterbox/normalize 上 TPU）
   - 硬解码：sophon-mw `bm_video_decode`（VPU 解码 → TPU 零拷贝）
3. 各 binding 层（CAPI/pybind/C#/Rust）同步暴露 SOPHGO 后端选项。

约束：目前无硬件、无 Linux 构建环境。所有 Sophgo 代码写入 `#ifdef ENABLE_SOPHGO` 块，保证 Windows 现有构建不受影响，编译验证后续在 Linux/设备上进行。

## 架构决策（已与用户确认）

- **算子抽象，不是图像容器抽象**：抽象"前后处理算子"（processor backend），不抽象"图像容器"。`ImageData` 保持现状，继续作为 CPU 兼容层（pImpl 包 cv::Mat）。
- **设备图像由各 backend 内部私有管理**：`SophgoProcessorBackend` 内部持有 `sail::BMImage`，`CudaProcessorBackend` 内部持有 `cuda::GpuMat`，对外接口统一收 CPU `ImageData`。
- **零拷贝窄口子**：processor backend 提供 `process_device_image(void* device_image, ...)`，仅硬解码路径使用。Sophgo 实现里 `device_image` 即 `sail::BMImage*`，跳过 CPU 上传。
- **后端选型混用**：推理用 SOPHON-Sail（高层 API），预处理用 BMCV，硬解码用 sophon-mw。Sail 负责省心的 tensor 管理，BMCV 保证预处理在 TPU 上执行。
- **backend 自动选择**：用工厂函数根据 `RuntimeOption.device + backend` 创建 processor backend，替代现在的布尔开关。

## 组件设计

### 1. `VisionProcessorBackend` 抽象接口

位置：`csrc/vision/processors/processor_backend.h`

```cpp
namespace modeldeploy::vision {

class LetterBoxRecord;   // 现有结构，前向声明

class VisionProcessorBackend {
public:
    virtual ~VisionProcessorBackend() = default;

    // 公共算子 —— 从现有 vision/common/processors 提取，签名对齐现有函数
    virtual bool letterbox(const ImageData& image, Tensor* out,
                           const std::vector<int>& dst_size, float pad_val,
                           LetterBoxRecord* record) = 0;
    virtual bool yolo_preprocess(const ImageData& image, Tensor* out,
                                 const std::vector<int>& dst_size, float pad_val,
                                 LetterBoxRecord* record) = 0;
    virtual bool yolo_preprocess_nv12(const uint8_t* y, const uint8_t* uv,
                                      const std::vector<int>& src_size,
                                      int step_y, int step_uv, Tensor* out,
                                      const std::vector<int>& dst_size,
                                      float pad_val, LetterBoxRecord* record) = 0;
    virtual bool resize(const ImageData& image, ImageData* out,
                        int width, int height) = 0;
    virtual bool normalize(const ImageData& image, ImageData* out,
                           const std::vector<float>& mean,
                           const std::vector<float>& std) = 0;
    virtual bool convert_to(const ImageData& image, ImageData* out,
                            const std::string& dst_format) = 0;
    virtual bool center_crop(const ImageData& image, ImageData* out,
                             int width, int height) = 0;
    virtual bool pad(const ImageData& image, ImageData* out,
                     const std::vector<int>& top, const std::vector<int>& bottom) = 0;
    virtual bool hwc2chw(const ImageData& image, Tensor* out) = 0;
    virtual bool normalize_and_permute(const ImageData& image, Tensor* out,
                                       const std::vector<float>& mean,
                                       const std::vector<float>& std) = 0;
    virtual bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                             int width, int height, ImageData* out) = 0;

    // 零拷贝窄口子：直接吃设备侧图像（硬解码路径专用）
    // device_image: Sophgo 实现为 sail::BMImage*；Cuda 实现为 cuda::GpuMat*；CPU 实现返回 false
    virtual bool process_device_image(void* device_image, int width, int height,
                                      Tensor* out, LetterBoxRecord* record) = 0;

    // 反推算子（TPU 上有加速 NMS，后续扩展）
    virtual bool nms(std::vector<DetectionResult>* results,
                     float conf_threshold, float nms_threshold) = 0;
};

} // namespace modeldeploy::vision
```

设计要点：
- 每个算子签名对齐现有 `vision/common/processors/*.h` 的函数签名，避免迁移成本。
- `ImageData*` 作为中间图像类型，让多算子 pipeline（resize→normalize→hwc2chw）能串联。CPU 实现直接用 cv::Mat，TPU 实现内部维护 BMImage 并在需要时与 ImageData 互转。
- `process_device_image` 默认返回 false（unsupported），只有支持设备图像的 backend 覆写。

### 2. 具体 backend 实现

位置：`csrc/vision/processors/` 下按设备分子目录

| 实现 | 路径 | 内容 |
|---|---|---|
| `CpuProcessorBackend` | `csrc/vision/processors/cpu/cpu_processor_backend.cpp` | 包装现有 `yolo_preprocess_cpu`、resize/normalize 等 OpenCV 实现 |
| `CudaProcessorBackend` | `csrc/vision/processors/cuda/cuda_processor_backend.cpp` | 包装现有 `yolo_preprocess_cuda`，`#ifdef WITH_GPU` 保护 |
| `SophgoProcessorBackend` | `csrc/vision/processors/sophgo/sophgo_processor_backend.cpp` | BMCV + `sail::BMImage`，`#ifdef ENABLE_SOPHGO` 保护 |

Sophgo 实现要点：
- 持有 `sail::Handle`（按 device_id 创建），用于 BMCV 图像操作。
- `letterbox`/`yolo_preprocess`：`ImageData`(CPU cv::Mat) → 上传为 `sail::BMImage` → `bmcv_image_resize`/`bmcv_image_convert_to`/letterbox → 输出 `sail::Tensor`（或拷回 CPU Tensor）。
- NV12 路径：硬解码出来的 NV12 buffer 直接 `bmcv_image_attach` 进 `sail::BMImage`，零拷贝。
- `process_device_image`：直接吃 `sail::BMImage*`，用于硬解码零拷贝。

### 3. backend 工厂 + 选择

位置：`csrc/vision/processors/processor_factory.h/.cpp`

```cpp
std::unique_ptr<VisionProcessorBackend> create_processor_backend(
    Device device, Backend backend, int device_id);
```

选择逻辑：
- `Device::CPU` → `CpuProcessorBackend`
- `Device::GPU` + `Backend::ORT/TRT/MNN` → `CudaProcessorBackend`（WITH_GPU 开启时）
- `Device::TPU`(新增) + `Backend::SOPHGO` → `SophgoProcessorBackend`
- 默认兜底 `CpuProcessorBackend`

### 4. 设备枚举扩展

`csrc/core/enum_variables.h`：
- `Backend` 增加 `SOPHGO`
- `Device` 增加 `TPU`
- `backend_to_string`/`device_to_string` 同步

### 5. RuntimeOption 扩展

`csrc/runtime/runtime_option.h/.cpp`：
- `void use_sophgo_backend(int device_id = 0);` — 设置 `backend = SOPHGO`, `device = TPU`, `device_id`
- `SophgoBackendOption` 结构：`{ int device_id; std::string bmodel_path; }`（bmodel_path 默认取 `model_file`）
- 新增 `runtime/backends/sophgo/option.h`

### 6. 推理后端 `SophgoBackend`

位置：`csrc/runtime/backends/sophgo/sophgo_backend.h/.cpp`

```cpp
class SophgoBackend : public BaseBackend {
public:
    bool init(const RuntimeOption& option) override;   // sail::Engine 加载 .bmodel
    bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;
    std::unique_ptr<BaseBackend> clone(...) override;
    size_t num_inputs() const override;
    size_t num_outputs() const override;
    TensorInfo get_input_info(int index) override;
    TensorInfo get_output_info(int index) override;
    // ...
private:
    sail::Engine engine_;
    std::string graph_name_;
    std::vector<TensorInfo> inputs_desc_;
    std::vector<TensorInfo> outputs_desc_;
};
```

实现要点：
- `init`：`sail::Engine` 加载 `.bmodel`，读取 graph 的输入/输出名、形状、dtype 填充 `inputs_desc_/outputs_desc_`。
- `infer`：输入 `Tensor`(CPU) → 拷入 `sail::Tensor`（或直接复用 device tensor）→ `engine.process()` → 输出 `sail::Tensor` → 拷回 CPU `Tensor`。
- `clone`：加载同一 bmodel 创建新 `sail::Engine`（多路推理时每路独立 engine，与现有 clone 语义一致）。
- `Runtime::init` 增加 `create_sophgo_backend()` 分支，`#ifdef ENABLE_SOPHGO` 保护。

### 7. CMake 集成

`CMakeLists.txt`：
- 新增 `option(ENABLE_SOPHGO "enable sophgo backend" OFF)`
- `if (ENABLE_SOPHGO)` → `add_definitions(-DENABLE_SOPHGO)`，`include(cmake/sophgo.cmake)`，`list(APPEND ALL_SOURCE ${SOPHGO_BACKEND_SOURCE})`
- `cmake/sophgo.cmake`：`find_path(SOPHON_SDK_DIR)` 查找 sail 头文件/库，找不到时静默禁用（对齐 OpenSSL 的做法），仅在 Linux 上启用（`if (NOT WIN32)`）

### 8. preprocessor 改造

11 个模型 preprocessor（classification/detection/iseg/obb/pose/face_*/lpr_*）：
- 移除 `use_cuda_preproc_` 布尔开关，改为持有 `std::shared_ptr<VisionProcessorBackend> backend_`
- `preprocess()` 内部直接调用 `backend_->yolo_preprocess(...)` 等算子
- backend 在 `BaseModel::initialize()`/`init_runtime()` 时通过工厂创建并注入 preprocessor

### 9. 硬解码（阶段 3）

`application/` 下 surveillance 应用的解码模块接入 sophon-mw `bm_video_decode`：
- 解码输出 NV12 `bm_image` → 直接走 `SophgoProcessorBackend::process_device_image` 零拷贝进 TPU
- 与现有 ffmpeg 解码路径并存，按设备自动选择
- 此阶段依赖真机，代码先写好 + `#ifdef ENABLE_SOPHGO` 保护

### 10. Binding 层（阶段 4）

- CAPI：`MDRuntimeOption` 增加 `use_sophgo_backend` 字段
- pybind：`RuntimeOption` 增加 `use_sophgo_backend()`
- C#：`RuntimeOption` 增加 `UseSophgoBackend()`
- Rust：`RuntimeOption` 增加 `use_sophgo_backend()` + FFI 同步

## 数据流

```
推理链路（现有模型）：
  ImageData(CPU) → [preprocessor] → Tensor → [SophgoBackend::infer] → Tensor → [postprocessor] → 结果
                      │                                       │
                      └── SophgoProcessorBackend (BMCV)        └── sail::Engine (.bmodel)

硬解码链路（阶段3）：
  RTSP流 → bm_video_decode(VPU) → BMImage(NV12) ──process_device_image──> TPU 推理
                                    │ 零拷贝
                                    └──> SophgoProcessorBackend
```

## 实施阶段

| 阶段 | 内容 | 可交付 |
|---|---|---|
| P1 | VisionProcessorBackend 抽象 + CPU/CUDA 实现 + 工厂 + 11 个 preprocessor 改造 + Device/Backend 枚举 + RuntimeOption | Windows 现有构建 + 全部模型测试通过，无行为变化 |
| P2 | SophgoBackend(推理) + SophgoProcessorBackend(BMCV 预处理) + CMake | `ENABLE_SOPHGO=ON` 代码完整，Linux 上可编译 |
| P3 | 硬解码（sophon-mw）接入 surveillance | 真机验证 |
| P4 | 各 binding 层暴露 SOPHGO | CAPI/pybind/C#/Rust 编译通过 |

P1 不依赖硬件，本仓库可完整验证。P2 代码完整但需 Linux 编译验证。P3/P4 依赖真机/后续环境。

## 测试策略

- P1：现有 Catch2 测试全绿（`[core]` + `[vision_models]`），确保抽象重构零行为变化；新增 processor factory 单元测试
- P2：新增 sophgo 编译期验证（Linux CI 可选）；新增 demo（`examples/demo_det` 增加 `demo_detection_sophgo.cpp`）
- P3：真机集成测试
- P4：各 binding 层基础编译检查

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| SOPHON-Sail 无 Windows 支持，无法本机编译 | `#ifdef ENABLE_SOPHGO` + CMake 开关隔离，Windows 构建完全不受影响 |
| sail API 版本差异（BM1688 vs BM1684X SDK） | 代码只依赖稳定核心 API（sail::Engine/Tensor/BMImage），版本适配留在编译期 |
| 无法真机验证 BMCV 行为 | 保留 CPU 兜底路径，`create_processor_backend` 在 sophgo 初始化失败时回退 CPU |
| 硬解码 API（bm_video_decode）与软件解码行为差异 | 解码统一封装 `DecodeBackend` 接口，硬解/软解可切换 |

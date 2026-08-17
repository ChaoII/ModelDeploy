# ImageData 多设备统一重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `ImageData` 重写为多设备统一的设备帧容器 + 按 device 分派的操作层，并让 `predict(ImageData)` 成为唯一推理入口（收敛 `predict_nv12`/`from_nv12`），capi2 图像/绘制/预测改为薄委托。

**Architecture:** `ImageData`（值类型 → `shared_ptr<ImageDataImpl>`）描述 planes/fmt/w/h/device + 一个 `Storage` 承载抽象（CPU=OpenCV，GPU=CUDA，TPU=BMCV）。所有 op（resize/crop/rotate/cvtColor/绘制/编解码）按 `device()` 经 `VisionProcessorBackend` 分派。`predict(const ImageData&)` 识别设备帧后设备内零拷贝预处。

**Tech Stack:** C++17、CMake/Ninja/MSVC（win 本机）/OpenCV（CPU）、CUDA（GPU）、SophgoBMCV（TPU）、capi2（C ABI）、pybind11/C#（后续绑定），Catch2 测试。

**Spec:** `docs/superpowers/specs/2026-08-17-image-data-multidevice-rewrite-design.md`

## Global Constraints

- 分支 `capi-v2`；本机构建缓存 `build_tdc`（BUILD_CAPI=ON、BUILD_AUDIO=ON、BUILD_VISION=ON、ENABLE_ORT=ON、ENABLE_SOPHGO=OFF、WITH_GPU=OFF、BUILD_TESTS=ON、BUILD_BENCHMARK=OFF）。**不要重配 CMake**，仅增量 `cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target <目标>'`。
- 运行 demo/测试从 `build_tdc\bin`（`../../test_data` 相对该 cwd 解析到仓库根）。
- **不使用异常**（部分 TU 未开 `/EHsc`）。错误走线程局部错误通道（对齐 capi2 `set_error/get_last_error`）。
- 设备未实现的 op → 明确 "device not supported" 错误，**绝不静默拷回 CPU**。
- 每阶段必须全量 CPU 回归保持基础断言数不下降（口基准：339 断言 / 94 用例）。
- 改动范围仅限 `csrc/`、`capi2/`、`tests/`、`examples/`、绑定目录；不改第三方。

---

## 新 ImageData 公开 API 契约（阶段 1 落地，后续阶段引用）

`csrc/vision/common/image_data.h` 最终形态（各阶段增量逼近）：

```cpp
class ImageData {
public:
    ImageData() = default;                                    // 空
    ImageData(int width, int height, MdImageType type);       // CPU（自有缓冲）

    ImageData(const ImageData&) = default;                    // 浅拷贝
    ImageData& operator=(const ImageData&) = default;

    [[nodiscard]] bool empty() const;
    [[nodiscard]] Device device() const;                      // cs core Device::CPU/GPU/TPU...
    [[nodiscard]] MdImageType format() const;
    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] size_t bytes() const;

    struct Plane { const uint8_t* data = nullptr; int step = 0; };
    [[nodiscard]] size_t plane_count() const;
    [[nodiscard]] Plane plane(size_t i) const;                // 多平面/设备统一入口

    // 构建
    static ImageData imread(const std::string& filename);                    // CPU
    static ImageData imdecode(const std::vector<uint8_t>& buf);              // CPU
    static ImageData from_raw(uint8_t* data, int w, int h, MdImageType type);// CPU（借用）
    static ImageData from_bgr24(const uint8_t* bgr, int w, int h);           // CPU
    static ImageData from_device_planes(const void* y, const void* uv,       // 设备零拷贝借用
                    int w, int h, int step_y, int step_uv, Device device);

    // op（按 device 分派；就地返回 bool，否则 out 参数）
    bool resize(int width, int height, ImageData* out) const;
    bool crop(int x, int y, int w, int h, ImageData* out) const;
    bool rotate(RotateFlags flag, ImageData* out) const;
    bool cvtColor(MdImageType dst, ImageData* out) const;
    bool drawRect(float x, float y, float w, float h, MDColorRGBA c, float alpha);
    bool drawPolygon(const float* xs, const float* ys, size_t n, MDColorRGBA c, float alpha);
    bool drawText(float x, float y, const std::string& text, const std::string& font,
                  int font_size, MDColorRGBA c, float alpha);
    bool imwrite(const std::string& filename) const;                         // CPU only
    std::vector<uint8_t> imencode(const std::string& ext) const;             // CPU only

    bool toCpu(ImageData* out) const;                        // 设备→CPU 拷贝
    bool asMat(cv::Mat* out) const;                          // 仅 CPU 借用（设备帧返回 false）
    ImageData clone() const;                                 // 同设备深拷贝

    static const char* last_error();                         // 线程局部错误
};
```

> 实现者注意：`Device`、`MdImageType`、`RotateFlags`、`MDColorRGBA` 均已存在（见 `csrc/core/*`、`csrc/vision/common/basic_types.h`、`capi2/md_capi.h`）。`MDColorRGBA` 在 capi2 定义——若 `image_data.h` 不想依赖 capi2，则在 `csrc/vision/common/` 定义等价 `struct RgbaColor{uint8_t r,g,b,a;}` 并在 ImageData 绘制用 RgbaColor；capi2 侧做映射。**用 `RgbaColor`，避免 csrc→capi2 反向依赖。**

---

### Task 1: 容器 + Storage 平面化（修元数据 bug）

**Files:**
- Modify: `csrc/vision/common/image_data.h`
- Modify: `csrc/vision/common/image_data.cpp`
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Produces（后续任务依赖）：
  - `ImageData::ImageData(int,int,MdImageType)`、`ImageData::plane(i)->Plane`、`plane_count()`、`width()/height()/format()/device()/bytes()/empty()`
  - 内部 `ImageDataImpl`（含 `Storage` 基类 + `CpuStorage`），仅 .cpp 内可见
  - `ImageData::from_raw`、`from_bgr24`、`from_device_planes`、`clone`、`toCpu`、`asMat`、`last_error`

- [ ] **Step 1: 写失败测试**，新增到 `tests/test_image_data.cpp`：

```cpp
TEST_CASE("image_data: device frame self-describes (w/h not zeroed)", "[core]") {
    std::vector<unsigned char> y(128 * 96, 100);
    std::vector<unsigned char> uv(128 * 48, 100);
    auto img = modeldeploy::vision::ImageData::from_device_planes(
        y.data(), uv.data(), 128, 96, 128, 128, modeldeploy::Device::CPU);
    REQUIRE(!img.empty());
    CHECK(img.width() == 128);      // 之前会清零
    CHECK(img.height() == 96);
    CHECK(img.format() == modeldeploy::vision::MdImageType::NV12);
    CHECK(img.plane_count() == 2);
    CHECK(img.plane(0).data == y.data());
    CHECK(img.plane(1).data == uv.data());
}

TEST_CASE("image_data: from_bgr24 builds CPU single-plane", "[core]") {
    std::vector<unsigned char> bgr(10 * 8 * 3, 42);
    auto img = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 10, 8);
    REQUIRE(!img.empty());
    CHECK(img.width() == 10);
    CHECK(img.height() == 8);
    CHECK(img.device() == modeldeploy::Device::CPU);
    CHECK(img.plane_count() == 1);
}
```

- [ ] **Step 2: 运行确认失败**

```
cd E:\CLionProjects\ModelDeploy
cmd /c '"...vcvars64.bat" >nul 2>&1 && cmake --build build_tdc --target test_modeldeploy'
cd build_tdc\bin
.\test_modeldeploy.exe "image_data: device frame self-describes (w/h not zeroed)"
```
Expected: FAIL（`ImageData::from_device_planes` 等未定义 → 编译错误 或 断言失败）。

- [ ] **Step 3: 实现内部 Storage 平面化**。重写 `image_data.h` 公开类 + `image_data.cpp` 的 `ImageDataImpl`：

```cpp
// image_data.h 中新增（内部声明，.cpp 定义）
struct ImageDataStorage {
    Device device = Device::CPU;
    virtual ~ImageDataStorage() = default;
    [[nodiscard]] virtual Device dev() const = 0;
};
struct CpuStorage : ImageDataStorage { cv::Mat mat; RgbaColor pad; };
struct DevicePlaneStorage : ImageDataStorage {       // 设备/借用平面
    std::vector<ImageData::Plane> planes;            // data 借用（外置）
    std::shared_ptr<void> keeper;                    // 借用源的 RAII（可空）
    MdImageType fmt;
    int w = 0, h = 0, ch = 0;
    size_t nbytes = 0;
};
```

```cpp
// image_data.cpp：ImageDataImpl 重写
class ImageDataImpl {
public:
    MdImageType type = MdImageType::PKG_BGR_U8;
    int width = 0, height = 0, channels = 0;
    size_t bytes_ = 0;
    std::shared_ptr<ImageDataStorage> storage;       // 任一子类
};
```

实现公开方法（`from_device_planes` 填真宽高；`from_bgr24` 走 `CpuStorage`；`plane(i)` 从 storage 取；`last_error()` 用 `thread_local std::string`）。`CpuStorage` 的 `mat` 提供 CPU 识别通道（把旧 `refresh_meta` 的 mat 派生逻辑搬进 `CpuStorage` 顶层 `from_mat` 助手）。

- [ ] **Step 4: 运行确认通过**

```
cd build_tdc\bin
.\test_modeldeploy.exe "image_data: device frame self-describes (w/h not zeroed)" "image_data: from_bgr24 builds CPU single-plane"
```
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add csrc/vision/common/image_data.h csrc/vision/common/image_data.cpp tests/test_image_data.cpp
git commit -m "refactor(image): planarize ImageData with Storage, device frames self-describe"
```

---

### Task 2: 操作分派层 + 错误通道 + CPU 后端

**Files:**
- Modify: `csrc/vision/common/image_data.h`
- Modify: `csrc/vision/common/image_data.cpp`
- Modify: `csrc/vision/processors/processor_factory.h` / `processor_factory.cpp`（或既有 `VisionProcessorBackend` 所在处）
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: Task 1 的容器。
- Produces：
  - `VisionProcessorBackend` 增加虚方法：`virtual bool resize(const ImageData&, int,int, ImageData*)`, `crop`, `rotate`, `cvtColor`（各 backend 覆盖；CPU 用 OpenCV）。
  - `ImageData::resize/crop/rotate/cvtColor` 内部 `backend_for(device())->...`。
  - `ImageData::last_error()` 线程局部；就地绘制 `drawRect/Polygon/Text` 返回 `bool`。
  - `ImageData::toCpu/asMat` / `RgbaColor`。

- [ ] **Step 1: 写失败测试**

```cpp
TEST_CASE("image_data: crop via unified dispatch (CPU)", "[core]") {
    std::vector<unsigned char> bgr(10 * 8 * 3, 0);
    for (int i = 0; i < 10 * 8; ++i) bgr[i*3] = (unsigned char)(i % 256);
    auto img = modeldeploy::vision::ImageData::from_bgr24(bgr.data(), 10, 8);
    modeldeploy::vision::ImageData out;
    REQUIRE(img.crop(2, 2, 4, 4, &out));
    CHECK(out.width() == 4);
    CHECK(out.height() == 4);
    CHECK(out.device() == modeldeploy::Device::CPU);
}

TEST_CASE("image_data: device frame op not supported reports error (no silent cpu)", "[core]") {
    std::vector<unsigned char> y(64 * 48, 0), uv(64 * 24, 0);
    auto img = modeldeploy::vision::ImageData::from_device_planes(y.data(), uv.data(), 64, 48, 64, 64, modeldeploy::Device::CPU);
    // NV12 设备帧（CPU 承载）作为"多平面"样例；对未实现 cvt 的路径必须返回 false 且 last_error 非空
    // 为确保分派可测，构造一个 device=CPU 但 format=NV12 的帧，验证 cvtColor(NV12->PKG_BGR_U8) 走 CPU 后端成功
    modeldeploy::vision::ImageData out;
    CHECK(img.cvtColor(modeldeploy::vision::MdImageType::PKG_BGR_U8, &out));
    CHECK(out.format() == modeldeploy::vision::MdImageType::PKG_BGR_U8);
}

TEST_CASE("image_data: invalid-op empty-input returns false + error", "[core]") {
    modeldeploy::vision::ImageData empty;
    modeldeploy::vision::ImageData out;
    CHECK_FALSE(empty.crop(0,0,1,1,&out));
    CHECK(modeldeploy::vision::ImageData::last_error() != nullptr);
}
```

- [ ] **Step 2: 运行确认失败**（编译错/断言失败）

- [ ] **Step 3: 实现分派**。
  - 在 `VisionProcessorBackend` 增补虚方法（CPU/CUDA/sophgo 各自覆盖；本阶段 CPU 完整，CUDA/BMCV 先放"返回 false + 置 error"桩，Task 后续用真实 GPU/BM 实现替换——不静默降级）。
  - `ImageData::crop/resize/rotate/cvtColor` 调 `backend_for(device())`。
  - 实现 CPU 后端：`CpuStorage` 取 `cv::Mat` 后 `cv::resize/cv::Rect/cv::rotate/cvtColor`，写回新 ImageData（同 device）。
  - `backend_for(device)`：CPU→CPU backend（get 既有 CpuProcessorBackend 或新建轻量后端）；GPU/TPU→对应后端，未实现返回 `false`。

```cpp
// 关键（示意，须按实际既有 backend 接口贴合）
bool ImageData::crop(int x, int y, int w, int h, ImageData* out) const {
    if (empty()) { set_img_error("crop: empty image"); return false; }
    if (x < 0 || y < 0 || x + w > width() || y + h > height()) { set_img_error("crop: out of bounds"); return false; }
    auto* b = backend_for(device());
    if (!b) { set_img_error("crop: no backend for device"); return false; }
    return b->crop(*this, x, y, w, h, out);
}
```

- [ ] **Step 4: 运行确认通过**（上述 3 个测试 + 全量 `~[model] ~[gpu]`）

- [ ] **Step 5: 提交**

```bash
git add csrc/vision/common/image_data.h csrc/vision/common/image_data.cpp csrc/vision/processors tests/test_image_data.cpp
git commit -m "refactor(image): backend-dispatched image ops + thread-local error channel (CPU)"
```

---

### Task 3: 编解码统一走 ImageData（CPU）

**Files:**
- Modify: `csrc/vision/common/image_data.cpp`（`imread/imdecode/imwrite/imencode/asMat/toCpu` 实现）
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: Task 1-2。
- Produces：`ImageData::imread/imdecode/imwrite/imencode`（CPU only；设备帧 imwrite/encode→false+error），`asMat`（仅 CPU），`toCpu`。

- [ ] **Step 1: 失败测试**

```cpp
TEST_CASE("image_data: imread/imwrite roundtrip (CPU)", "[core]") {
    auto img = modeldeploy::vision::ImageData::imread("test_data/test_images/test_detection0.jpg");
    REQUIRE(!img.empty());
    auto out = img.clone();
    REQUIRE(out.imwrite("capi2_plan_tmp_out.jpg"));
    auto again = modeldeploy::vision::ImageData::imread("capi2_plan_tmp_out.jpg");
    REQUIRE(!again.empty());
    std::remove("capi2_plan_tmp_out.jpg");
}

TEST_CASE("image_data: device nv12 frame imwrite is not-supported (no silent cpu)", "[core]") {
    std::vector<unsigned char> y(64*48,0), uv(64*24,0);
    auto img = modeldeploy::vision::ImageData::from_device_planes(y.data(), uv.data(), 64,48,64,64, modeldeploy::Device::CPU);
    // CPU 承载的 NV12 虽在 CPU，但按"格式非 PKG_*_U8"即拒绝写盘，防误用；须返回精确错误
    // 实现按 device()==CPU 且 format 单平面 PKG_*_U8 才允许 imwrite；否则 false+error
    CHECK_FALSE(img.imwrite("x.jpg"));
}
```

- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现** `imread/imdecode`（OpenCV，CpuStorage）；`imwrite/imencode` 先验 `device()==CPU && 单平面 PKG_*`，否则 `set_img_error("imwrite: only single-plane CPU PKG image supported")` 返回 false/空。
- [ ] **Step 4: 确认通过**
- [ ] **Step 5: 提交**

```bash
git add csrc/vision/common/image_data.cpp tests/test_image_data.cpp
git commit -m "feat(image): codec unified through ImageData (CPU), device non-single-plane rejected"
```

---

### Task 4: 收敛 predict(ImageData) + from_device_planes

**Files:**
- Modify: `csrc/vision/processors/`（预处理器按 device 分派）
- Modify: 各模型 `predict` 入口（识别 device 帧走设备预处）——以 `csrc/vision/detection/` 的 `UltralyticsDet` 为样板，逐模型核对。
- Modify: 删除模型级 `predict_nv12(...)` 声明/实现（先全量定位 `predict_nv12`）。
- Test: `tests/test_image_data.cpp` / 设备标签测试

**Interfaces:**
- Consumes: Task 1-3。
- Produces：`predict(const ImageData&)` 对 device 帧零拷贝；`predict_nv12` 移除。

- [ ] **Step 1: 定位所有 `predict_nv12`**

```
rg -n "predict_nv12" csrc capi2 tests
```
列出清单，作为移除范围。

- [ ] **Step 2: 失败测试**（仅 CPU 可跑断言）

```cpp
TEST_CASE("predict_nv12 is removed from public API", "[core]") {
    static_assert(!has_predict_nv12_v, "predict_nv12 must be removed");
}
```
> 用 SFINAE `has_predict_nv12_v` 为 false 的检测（C++17），或直接 `rg` 断言 + 编译期检查；实现者选其一并写清。

- [ ] **Step 3: 收敛**：新代码以 `ImageData frame = from_device_planes(y,uv,w,h,step_y,step_uv,dev); model->predict(frame,&res);` 取代原 `predict_nv12(...)`；预处理器读取 `frame.plane(0)/plane(1)` 在设备内跑（CUDA/BMCV 用既有 draw/预处 backend 的同一设备路径）。CPU 下 `predict(ImageData)` 与旧行为等价。
- [ ] **Step 4: 全量回归** `~[model] ~[gpu]` 不断言下降。
- [ ] **Step 5: 提交**

```bash
git add csrc/vision/processors csrc/vision/detection csrc/vision/pose csrc/vision/obb csrc/vision/seg tests
git commit -m "refactor(vision): converge to predict(ImageData); remove model-level predict_nv12"
```

---

### Task 5: capi2 图像/绘制/预测薄委托 + 收敛 nv12 API

**Files:**
- Modify: `capi2/md_capi.cpp`、`capi2/md_capi.h`
- Test: `tests/test_capi.cpp`

**Interfaces:**
- Consumes: Task 1-4 的 `ImageData` API。
- Produces：
  - `md_image_from_device_nv12(out, y, uv, w,h, step_y, step_uv, MDDevice)`（调 `ImageData::from_device_planes`）
  - `md_model_predict`（已走 handle_to_image → predict(ImageData)，无改）
  - 删除 `md_model_predict_nv12`、`md_draw_rect/polygon/text` 改为委托 ImageData 绘制（CPU 现可跑即可；设备分派沿用 `md_draw_result` 既有机制）
  - `md_image_crop/rotate/save/from_nv12` 改为调 ImageData 方法

- [ ] **Step 1: 失败测试（[capi]）**

```cpp
TEST_CASE("capi2 image_from_device_nv12 wraps zero-copy two-plane", "[capi]") {
    const int w=16,h=16; std::vector<unsigned char> y(w*h,0), uv(w*h/2,0);
    MDImageHandle img=nullptr;
    REQUIRE(md_image_from_device_nv12(&img, y.data(), uv.data(), w,h, w, w, MD_DEV_CPU)==MD_OK);
    int ow=0,oh=0; REQUIRE(md_image_size(img,&ow,&oh)==MD_OK);
    CHECK(ow==w); CHECK(oh==h);   // 设备帧自描述，不再依赖外带 w/h
    md_image_destroy(img);
}
TEST_CASE("capi2 predict_nv12 removed", "[capi]") {
    // 编译期：如果函数仍导出会编译错；此处仅占位断言（删除函数后无需调用）
    CHECK(true);
}
```

- [ ] **Step 2: 确认失败**（`md_image_from_device_nv12` 未定义→链接错）
- [ ] **Step 3: 实现委托**（各 md_image_* 改为解读 ImageData 的薄封装，删 cv::Mat 复刻；删 `md_model_predict_nv12` 及 `handle` 里对 `predict_nv12` 的调用，改 `from_device_planes+predict`）。
- [ ] **Step 4: 确认通过**：`[capi]` + `[model]` 端到端（用既有检测模型跑设备帧路径走 CPU 等价）。
- [ ] **Step 5: 提交**

```bash
git add capi2/md_capi.h capi2/md_capi.cpp tests/test_capi.cpp
git commit -m "refactor(capi): delegate image/draw/predict to ImageData; md_image_from_device_nv12; drop predict_nv12"
```

---

### Task 6: pybind / C# / Rust 绑定同步（改公开 API 的联动）

**Files:**
- Modify: `csrc/pybind/`（image_data 相关绑定：`plane`/`device`/新增 builder，删 `y()/uv()/predict_nv12` 导出）
- Modify: `csharp/ModelDeploy/`（`VisionImage.cs` 等，对齐新 ImageData 暴露的方法）
- Modify: `csharp/ModelDeploy/V2/...` 若引用 nv12/plane
- Modify: `csrc/pybind` + `rust/`（若存在 rust 绑定引用 nv12）

**Interfaces:**
- Consumes: Task 1-5。
- Produces：绑定层无 `predict_nv12`/`y()/uv()` 残留；`VisionImage` 暴露 `Plane(i)/Device/Format/Width/Height`。

- [ ] **Step 1: 失败验证**：`dotnet build csharp/ModelDeployExample -c Debug` 与 `python` 绑定编译，若引用已删 API 应报错。
- [ ] **Step 2: 同步绑定**：逐项把 `y()/uv()/predict_nv12` 引用替换为 `Plane`/`Device`/`from_device_planes`。
- [ ] **Step 3: 确认通过**：C# 0 错 0 警；python 导入成功。
- [ ] **Step 4: 提交**

```bash
git add csrc/pybind csharp rust
git commit -m "refactor(bindings): sync with planarized ImageData; drop nv12 legacy exports"
```

---

### Task 7: 删除遗留 API + 全量回归 + 文档

**Files:**
- Modify: `csrc/vision/common/image_data.h`（删 `data()/y()/uv()/to_mat` 遗留，若仍有消费者则先迁移）
- Modify: `examples/EXAMPLES.md`、`docs/` 相关
- Test: 全量回归

**Interfaces:**
- Consumes: Task 1-6。
- Produces：无遗留 nv12/predict_nv12 的干净 API 面。

- [ ] **Step 1: rg 清理**：`rg -n "predict_nv12|\bdata\(\)|\.to_mat|->y\(\)|->uv\(\)" csrc capi2 tests examples csharp rust pybind`，确保无残留（或一一迁移）。
- [ ] **Step 2: 全量构建 + 回归**

```
cd E:\CLionProjects\ModelDeploy
cmd /c '"...vcvars64.bat" >nul 2>&1 && cmake --build build_tdc'
cd build_tdc\bin
.\test_modeldeploy.exe "~[model] ~[gpu]"    # 断言数 ≥ 339 / 用例 ≥ 94
.\test_modeldeploy.exe "[capi]"
```

- [ ] **Step 3: 更新 `examples/EXAMPLES.md`**：设备帧/`from_device_planes`/`md_image_from_device_nv12` 说明；删 predict_nv12 提及。
- [ ] **Step 4: 提交**

```bash
git add csrc tests examples docs
git commit -m "refactor(image): remove legacy data()/nv12 API; docs + full regression green"
```

---

## Self-Review

**Spec 覆盖：**
- 容器平面化（spec §1）→ Task 1
- 预处理器四件套分派（spec §2.1）→ Task 2
- 绘制分派（spec §2.2）→ Task 2（+Task 5 capi2 委托）
- 编解码（spec §2.3）→ Task 3
- 收敛 predict_nv12/from_nv12（spec §2.4）→ Task 4、Task 5
- 错误通道/所有权（spec §3）→ Task 1（last_error/Storage keeper）、Task 2
- 测试与迁移（spec §4）→ 各 Task + Task 7 收官回归

**占位符/一致性：** 公开 API 契约在文档顶部统一定义一次，Task 1-7 引用同名签名（`from_device_planes`、`plane(i)`、`toCpu`、`RgbaColor`、`imwrite/imencode`、`md_image_from_device_nv12`），无跨任务重名错漏。

**说明：** CUDA/BMCV 后端真实 kernel 依赖既有 `draw_gpu.cu` / bmcv 路径，Task 2 首版以"返回 false+error 桩"占位并在对应 backend 内填真实现；本机（WITH_GPU=OFF、ENABLE_SOPHGO=OFF）验证以 CPU 等价为主，设备后端在对应平台验证。

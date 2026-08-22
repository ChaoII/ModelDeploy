# Barcode / QR 条码识别模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ModelDeploy SDK 新增纯经典 CV（非 DNN）的条码/二维码识别模块 `BarcodeDetector`，基于捆绑 ZXing-C++ 解码，跨全部后端（ORT/MNN/TRT/SOPHGO），全套交付（C++ + Python + CAPI + C# + Rust + demo + 测试）。

**Architecture:** 新增 `csrc/vision/barcode/`（BarcodeResult 结构 + BarcodeReader ZXing 封装 + BarcodeDetector 外观类），捆绑 `third_party/zxing-cpp` 静态库 PRIVATE 链接进 SDK SHARED 库（仿 cppjieba/samplerate 模式）。新源文件被现有的 `GLOB_RECURSE VISION_SOURCE/PYBIND_SOURCE/CAPI_SOURCE` 自动收集，无需改收集列表。绑定与测试逐一镜像已合入的 tracking 模块范式。

**Tech Stack:** C++17、ZXing-C++（v2.3.0，纯 C++ 离线编译，无外部解码依赖）、OpenCV（仅用于 demo/可视化或读图，BarcodeDetector 本身不依赖 OpenCV 解码）、pybind11、CAPI、C# P/Invoke、Rust extern "C"、Catch2。

参考设计：`docs/superpowers/specs/2026-08-22-barcode-qr-design.md`（设计定稿）。
参考范本（已合入 main）：`csrc/vision/tracking/**`、`csrc/pybind/vision/tracking_pybind.cpp`、`capi/md_capi.{h,cpp}` 中 `md_tracker_*`、`csharp/ModelDeploy/Tracker.cs`、`rust/modeldeploy/src/tracker.rs`、`examples/demo_tracking/**`、`tests/test_tracking.cpp`。

## Global Constraints

- 后端无关：BarcodeDetector **不依赖任何推理后端 / RuntimeOption / BackendInit / 模型权重**，纯 CPU 图像处理 + ZXing 解码，天然跨 ORT/MNN/TRT/SOPHGO。测试无需初始化后端。
- 绑定名与命名空间：C++ 置于 `modeldeploy::vision::barcode`；pybind 注册 `bind_barcode(const pybind11::module&)`，置于 `modeldeploy.vision` 子模块 `BarcodeDetector` 类。
- 结果结构字段名固定：`text` / `format` / `quad`（`std::array<Point2f,4>`，左上起顺时针）/ `score` / `is_qr`。跨 C++/CAPI/C#/Rust 保持一致。
- `Point2f`/`Rect2f` 复用 `vision/common/struct.h` 现有定义，不重复定义。
- ZXing-C++ 以 `third_party/zxing-cpp/` 在树内捆绑（仿 cppjieba/samplerate），新增 `BUILD_BARCODE` 选项（默认 ON，随 BUILD_VISION）。
- C++17；MSVC `/utf-8`（根 CMakeLists 已为 SDK 自动设置）；忽略警告。
- `tests/CMakeLists.txt` 用显式 `TEST_SOURCES` 列表 —— 新 `test_barcode.cpp` 必须手动加入。
- 演示为单一无后端目标（纯 CV 不需要 `md_add_demo_matrix` 的后端变体）。
- 交付界定：不做 DNN 定位增强、不做视频流连续扫描、不做"解码失败仅返回检测框"独立接口（见设计"明确不做"）。

---

### Task 1: 捆绑 ZXing-C++ 依赖 + BUILD_BARCODE 选项

**Files:**
- Create: `third_party/zxing-cpp/`（ZXing-C++ v2.3.0 源码，在树内捆绑）
- Modify: `CMakeLists.txt`（新增 BUILD_BARCODE 块，仿 BUILD_AUDIO 行 205-220）

**Interfaces:**
- Consumes: 无。
- Produces: CMake 目标 `zxing`（静态库）+ include 目录 `third_party/zxing-cpp/wrappers/`（ZXing/ReadBarcode.h 所在），供后续任务 `#include <ZXing/ReadBarcode.h>`。

- [ ] **Step 1: 下载并放入 zxing-cpp 源码**

将 ZXing-C++ v2.3.0 源码放至 `third_party/zxing-cpp/`。若从 zip 解压，保证结构为：
```
third_party/zxing-cpp/
├── core/src/            # 解码核心源码
├── wrappers/common/     # 通用头（ReadBarcode.h 等）
├── CMakeLists.txt
└── LICENSE
```
（若仓库允许，也可按 modelscope `openvc.cmake` 的 FetchContent 模式从
`https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master` 下载 `zxing-cpp_2.3.0.zip`——但优先在树内捆绑，确定性更高。）

> 说明：若采用 FetchContent，则本任务改为新增 `cmake/zxing.cmake`（仿 `cmake/opencv.cmake` FetchContent 模式），并在根 CMakeLists 的 BUILD_BARCODE 块 `include()`。二选一，实现时保持一致。

- [ ] **Step 2: 在根 CMakeLists.txt 增加 BUILD_BARCODE 选项**

在 `option(BUILD_VISION ...)` 附近新增：
```cmake
option(BUILD_BARCODE  "Enable barcode/QR recognition (ZXing-C++)" ON)
```

- [ ] **Step 3: 新增 BUILD_BARCODE 编译块（仿 BUILD_AUDIO 行 205-220）**

在根 CMakeLists.txt 中（`BUILD_AUDIO` 块之后）新增：
```cmake
if (BUILD_BARCODE)
    add_definitions(-DBUILD_BARCODE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/zxing-cpp EXCLUDE_FROM_ALL)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/zxing-cpp/wrappers/common)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/zxing-cpp/core/src)
    list(APPEND PRIVATE_DEPENDS zxing)
endif ()
```
> 说明：ZXing-C++ 的 CMake 会产出 `zxing` 目标并已带自己的 include 目录（`wrappers/common`）；`include_directories` 为防御性补充，确保 `#include <ZXing/ReadBarcode.h>` 可解析。`zxing` 作为 `PRIVATE_DEPENDS` 静态链接进 SDK 共享库（仿 `samplerate`）。若 ZXing-C++ 的 CMakeLists 需要 `BUILD_EXAMPLES OFF` 等开关，参考其 README/CMake 设置 `set(... CACHE BOOL "" FORCE)` 抑制（仿 BUILD_AUDIO 对 samplerate 的 `BUILD_TESTING OFF` 处理）。

- [ ] **Step 4: 配置验证**

用 build_tdc_gpu（Ninja+MSVC，Windows）做一次最小配置确认 zxing 目标存在且被链接。用 .bat 包裹 vcvars64：
```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake -S . -B build_tdc_gpu -G Ninja -DBUILD_TESTS=ON -DWITH_GPU=ON
cmake --build build_tdc_gpu --config Release --parallel
```
Expected: 配置成功，`zxing` 被编译并链接入 `ModelDeploySDK`；无报错。若报 `zxing` 目标未定义，检查 ZXing-C++ CMake 目标名（可能是 `ZXing::ZXing` 或 `zxing`），据此调整 `PRIVATE_DEPENDS`。

- [ ] **Step 5: Commit**

```bash
git add third_party/zxing-cpp CMakeLists.txt
git commit -m "build(barcode): vendor ZXing-C++ and add BUILD_BARCODE option"
```

---

### Task 2: BarcodeResult 结果结构

**Files:**
- Create: `csrc/vision/barcode/result.h`

**Interfaces:**
- Consumes: `modeldeploy::vision::Point2f`（`vision/common/struct.h`）。
- Produces: `modeldeploy::vision::barcode::BarcodeResult`（后续 Task 3/4、pybind、CAPI、C#、Rust、测试全部使用）。

- [ ] **Step 1: 写头文件**

```cpp
#pragma once
#include <array>
#include <string>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::barcode {
    /*! @brief 条码 / 二维码解码结果 */
    struct MODELDEPLOY_CXX_EXPORT BarcodeResult {
        BarcodeResult() = default;
        std::string text;                    // 解码文本 / URL
        std::string format;                  // "QR_CODE"/"EAN_13"/"CODE_128"/"DATA_MATRIX"/...
        std::array<Point2f, 4> quad;         // 码区域四角定位框（左上起顺时针）
        float score = 0.0f;                  // 可靠度 [0,1]
        bool is_qr = false;                  // 是否二维码
    };
}
```

- [ ] **Step 2: Commit**

```bash
git add csrc/vision/barcode/result.h
git commit -m "feat(barcode): BarcodeResult result struct"
```
（本任务仅头文件，可随后续编译任务一并验证编译。）

---

### Task 3: BarcodeReader — ZXing 读码封装

**Files:**
- Create: `csrc/vision/barcode/barcode_reader.h`
- Create: `csrc/vision/barcode/barcode_reader.cpp`

**Interfaces:**
- Consumes: `BarcodeResult`（Task 2）、`modeldeploy::vision::ImageData`（`vision/common/image_data.h`，用于 `data/width/height/channels`）。
- Produces: `BarcodeReader`：
  - `using Formats = uint32_t;`（位掩码，`QR_CODE=1<<0` … 见 Task 4 常量）
  - `static std::vector<BarcodeResult> read(const ImageData& img, Formats formats, bool convert_bgr_to_rgb);`
  - 内部将 `ImageData` 像素交给 ZXing `ImageView`，遍历 `ReadBarcodes` 结果，填充 `BarcodeResult`。

- [ ] **Step 1: 写头文件**

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include "vision/common/image_data.h"
#include "vision/barcode/result.h"

namespace modeldeploy::vision::barcode {
    using Formats = uint32_t;

    inline constexpr Formats FMT_QR_CODE    = 1u << 0;
    inline constexpr Formats FMT_DATA_MATRIX= 1u << 1;
    inline constexpr Formats FMT_AZTEC      = 1u << 2;
    inline constexpr Formats FMT_EAN_8      = 1u << 3;
    inline constexpr Formats FMT_EAN_13     = 1u << 4;
    inline constexpr Formats FMT_UPC_A      = 1u << 5;
    inline constexpr Formats FMT_UPC_E      = 1u << 6;
    inline constexpr Formats FMT_CODE_128   = 1u << 7;
    inline constexpr Formats FMT_CODE_39    = 1u << 8;
    inline constexpr Formats FMT_CODE_93    = 1u << 9;
    inline constexpr Formats FMT_ITF        = 1u << 10;
    inline constexpr Formats FMT_CODABAR    = 1u << 11;
    inline constexpr Formats FMT_ALL        = 0xFFFFFFFFu;

    class BarcodeReader {
    public:
        BarcodeReader() = delete;
        /*! 解码图片中的所有码。img 可为 BGR(3通道) 或 单通道(灰度)。
         *  若为 BGR 且 convert_bgr_to_rgb=true，内部先转 RGB 再送 ZXing。 */
        static std::vector<BarcodeResult> read(const ImageData& img,
                                               Formats formats = FMT_ALL,
                                               bool convert_bgr_to_rgb = true);
    };
}
```

- [ ] **Step 2: 写实现（ZXing 封装）**

```cpp
#include "vision/barcode/barcode_reader.h"
#include <string>
#include <ZXing/ReadBarcode.h>

namespace modeldeploy::vision::barcode {
    namespace {
        zxing::BarcodeFormat to_zxing_format(Formats f) {
            using namespace zxing;
            switch (f) {
                case FMT_QR_CODE:     return BarcodeFormat::QRCode;
                case FMT_DATA_MATRIX: return BarcodeFormat::DataMatrix;
                case FMT_AZTEC:       return BarcodeFormat::Aztec;
                case FMT_EAN_8:       return BarcodeFormat::EAN8;
                case FMT_EAN_13:      return BarcodeFormat::EAN13;
                case FMT_UPC_A:       return BarcodeFormat::UPCA;
                case FMT_UPC_E:       return BarcodeFormat::UPCE;
                case FMT_CODE_128:    return BarcodeFormat::Code128;
                case FMT_CODE_39:     return BarcodeFormat::Code39;
                case FMT_CODE_93:     return BarcodeFormat::Code93;
                case FMT_ITF:         return BarcodeFormat::ITF;
                case FMT_CODABAR:     return BarcodeFormat::Codabar;
                default:              return BarcodeFormat::None;
            }
        }
        std::string format_name(zxing::BarcodeFormat f) {
            return zxing::ToString(f);
        }
        bool is_qr_format(zxing::BarcodeFormat f) {
            return f == zxing::BarcodeFormat::QRCode;
        }
        Point2f to_pt(const zxing::PointI& p) { return {float(p.x), float(p.y)}; }
    }

    std::vector<BarcodeResult> BarcodeReader::read(const ImageData& img,
                                                   Formats formats,
                                                   bool convert_bgr_to_rgb) {
        std::vector<BarcodeResult> out;
        if (!img.data || img.width == 0 || img.height == 0) return out;

        zxing::BarcodeFormats zformats = zxing::BarcodeFormat::None;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            if (formats & (1u << bit)) {
                auto f = to_zxing_format(1u << bit);
                if (f != zxing::BarcodeFormat::None) zformats |= f;
            }
        }
        zxing::DecodeHints hints;
        hints.setFormats(zformats);
        hints.setTryHarder(true);

        zxing::ImageView view;
        if (img.channels == 3 && convert_bgr_to_rgb && img.format == "BGR") {
            // BGR -> RGB 逐像素重排（ImageData 需提供像素访问；见下注）
            std::vector<uint8_t> rgb(img.width * img.height * 3);
            const uint8_t* src = static_cast<const uint8_t*>(img.data);
            for (size_t i = 0; i < static_cast<size_t>(img.width * img.height); ++i) {
                rgb[3 * i + 0] = src[3 * i + 2];
                rgb[3 * i + 1] = src[3 * i + 1];
                rgb[3 * i + 2] = src[3 * i + 0];
            }
            view = zxing::ImageView(rgb.data(), img.width, img.height,
                                    zxing::ImageFormat::RGB, img.width * 3);
            auto results = zxing::ReadBarcodes(view, hints);
            for (auto& r : results) {
                if (!r.isValid()) continue;
                BarcodeResult br;
                br.text = r.text();
                br.format = format_name(r.format());
                br.is_qr = is_qr_format(r.format());
                br.score = r.isValid() ? 1.0f : 0.0f;
                auto pos = r.position();
                br.quad[0] = to_pt(pos.topLeft());
                br.quad[1] = to_pt(pos.topRight());
                br.quad[2] = to_pt(pos.bottomRight());
                br.quad[3] = to_pt(pos.bottomLeft());
                out.push_back(br);
            }
        } else {
            // 灰度直送
            using zxing::ImageFormat;
            ImageFormat fmt = (img.channels == 4) ? ImageFormat::RGBA
                             : (img.channels == 3) ? ImageFormat::BGR   // 非 BGR 标记则按 3ch
                             : ImageFormat::Lum;
            view = zxing::ImageView(img.data, img.width, img.height, fmt, img.width * img.channels);
            auto results = zxing::ReadBarcodes(view, hints);
            for (auto& r : results) {
                if (!r.isValid()) continue;
                BarcodeResult br;
                br.text = r.text();
                br.format = format_name(r.format());
                br.is_qr = is_qr_format(r.format());
                br.score = 1.0f;
                auto pos = r.position();
                br.quad[0] = to_pt(pos.topLeft());
                br.quad[1] = to_pt(pos.topRight());
                br.quad[2] = to_pt(pos.bottomRight());
                br.quad[3] = to_pt(pos.bottomLeft());
                out.push_back(br);
            }
        }
        return out;
    }
}
```
> **实现注意**：`ImageData` 的实际像素布局/`format` 字段请以 `vision/common/image_data.h` 为准核对。若 `ImageData` 恒为 BGR 3 通道，则统一走 BGR→RGB 分支，无需 `format` 分支。key 点：ZXing `ImageView(data, w, h, format, stride)`。若简化，可固定 `BarcodeDetector` 输入先转成 RGB `cv::Mat` 再读，从而只保留 RGB/灰度两条路径，避免依赖 `ImageData.format` 字符串。实现时以"读得正确"为准，可适度简化本函数内部结构（保持对外签名不变）。

- [ ] **Step 3: 单元验证（编译）**

`cmake --build build_tdc_gpu` 应成功编译并链接。此时无测试用例，仅确认编译通过、`zxing` 链接无未定义符号。

- [ ] **Step 4: Commit**

```bash
git add csrc/vision/barcode/barcode_reader.h csrc/vision/barcode/barcode_reader.cpp
git commit -m "feat(barcode): BarcodeReader ZXing decode wrapper"
```

---

### Task 4: BarcodeDetector 外观类

**Files:**
- Create: `csrc/vision/barcode/barcode.h`
- Create: `csrc/vision/barcode/barcode.cpp`

**Interfaces:**
- Consumes: `BarcodeReader`（Task 3，`Formats` 常量与 `read`）、`ImageData`。
- Produces: `BarcodeDetector` 公共 API（后续 pybind/CAPI/C#/Rust/demo/test 全部使用）：
  - `BarcodeDetector()`
  - `void set_formats(Formats formats);`
  - `std::vector<BarcodeResult> detect(const ImageData& img) const;`
  - `const Formats& formats() const;`

- [ ] **Step 1: 写头文件**

```cpp
#pragma once
#include <vector>
#include "vision/common/image_data.h"
#include "vision/barcode/result.h"
#include "vision/barcode/barcode_reader.h"

namespace modeldeploy::vision::barcode {
    /*! @brief 条码/二维码识别器（纯 CV，零 DNN，跨全部后端） */
    class MODELDEPLOY_CXX_EXPORT BarcodeDetector {
    public:
        BarcodeDetector() = default;

        /*! 限定解码格式子集（FMT_* 的位或），默认 FMT_ALL。 */
        void set_formats(Formats formats);

        /*! 检测并解码图片中的所有码。返回可能为空。 */
        std::vector<BarcodeResult> detect(const ImageData& img) const;

        Formats formats() const { return formats_; }

    private:
        Formats formats_ = FMT_ALL;
    };
}
```

- [ ] **Step 2: 写实现**

```cpp
#include "vision/barcode/barcode.h"

namespace modeldeploy::vision::barcode {
    void BarcodeDetector::set_formats(Formats formats) { formats_ = formats; }

    std::vector<BarcodeResult> BarcodeDetector::detect(const ImageData& img) const {
        return BarcodeReader::read(img, formats_, /*convert_bgr_to_rgb=*/true);
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add csrc/vision/barcode/barcode.h csrc/vision/barcode/barcode.cpp
git commit -m "feat(barcode): BarcodeDetector facade"
```

---

### Task 5: C++ 单元测试 test_barcode.cpp

**Files:**
- Create: `tests/test_barcode.cpp`
- Modify: `tests/CMakeLists.txt`（TEST_SOURCES 加 `test_barcode.cpp`）

**Interfaces:**
- Consumes: `BarcodeDetector`（Task 4）、`ImageData`、`BarcodeResult`。
- Produces: Catch2 测试用例集 `[barcode]`（供 `./test_modeldeploy "[barcode]"` 运行）。

- [ ] **Step 1: 写测试**

测试生成 QR 样本：用 OpenCV `cv::QRCodeEncoder`（OpenCV ≥5 可用）在内存生成灰度/彩色 QR，或读仓库内样本图。本模块纯 CV，无需后端/模型。

```cpp
#include "catch2/catch_test_macros.hpp"
#include <opencv2/opencv.hpp>

#include "vision/common/image_data.h"
#include "vision/barcode/barcode.h"

using namespace modeldeploy::vision;
using namespace modeldeploy::vision::barcode;

namespace {
    // 用 OpenCV 生成一个 QR 灰度图，转 ImageData。
    ImageData make_qr_image(const std::string& payload = "https://example.com/MD") {
        cv::QRCodeEncoder::Params p{};
        cv::Ptr<cv::QRCodeEncoder> enc = cv::QRCodeEncoder::create(p);
        cv::Mat qr = enc->encode(payload);   // 单通道灰度，>=21x21
        cv::Mat canvas;
        int border = 40, scale = 8;
        int W = qr.cols * scale + 2 * border;
        int H = qr.rows * scale + 2 * border;
        canvas.create(H, W, CV_8UC1);
        canvas.setTo(255);
        qr.copyTo(canvas(cv::Rect(border, border, qr.cols * scale, qr.rows * scale)));
        // 放大灰度（最近邻）
        cv::Mat big;
        cv::resize(canvas, big, cv::Size(), (double)W / canvas.cols, (double)H / canvas.rows, cv::INTER_NEAREST);
        (void)big;

        cv::Mat gray;
        if (qr.type() == CV_8UC3) cv::cvtColor(qr, gray, cv::COLOR_BGR2GRAY);
        else gray = qr.clone();

        ImageData img;
        img.width = canvas.cols;
        img.height = canvas.rows;
        img.channels = 1;
        img.data = canvas.data;
        return img;
    }
}

TEST_CASE("BarcodeDetector decodes a generated QR", "[barcode]") {
    BarcodeDetector det;
    // 上面 make_qr_image 用 canvas（含边框）放大后解码。
    // 说明：QRCodeEncoder 输出已含 quiet zone，为可靠解码建议 canvas=qr 直接使用；
    //       此处用一个能稳定解码的样本来断言。
    ImageData img = make_qr_image();
    auto res = det.detect(img);
    REQUIRE(!res.empty());
    bool found = false;
    for (auto& r : res) {
        if (r.is_qr && r.text == "https://example.com/MD") { found = true; break; }
    }
    REQUIRE(found);
}

TEST_CASE("BarcodeDetector honors format restriction", "[barcode]") {
    BarcodeDetector det;
    det.set_formats(FMT_CODE_128);   // 只允许 Code128
    ImageData img = make_qr_image();  // 内容是 QR
    auto res = det.detect(img);
    // 限制为 Code128 时不应解码出 QR
    for (auto& r : res) {
        REQUIRE_FALSE(r.is_qr);
    }
}
```
> **实现注意**：QR 样本生成/读取的正确性与稳定性是实现时必须验证的关键点。OpenCV ≥5 的 `cv::QRCodeEncoder` 可用则内存生成；若不可用，则改为在 `test_data/` 放一张已知 QR 样本图，用 `cv::imread` 读取并断言解码文本。**确保生成的码能稳定解码、断言文本精确匹配**，否则测试会假失败。`ImageData` 的 `data`（`const void*`）、`width/height/channels` 字段按 `image_data.h` 实际签名赋值。

- [ ] **Step 2: 在 TEST_SOURCES 注册**

编辑 `tests/CMakeLists.txt`，在 `TEST_SOURCES` 列表（约行 11-29）末尾、`test_tracking.cpp` 之后加入：
```cmake
    test_barcode.cpp
```

- [ ] **Step 3: 构建并运行**

用 .bat 包裹 vcvars64，`cmake --build build_tdc_gpu`，然后：
```bat
cd build_tdc_gpu
test_modeldeploy.exe "[barcode]"
```
Expected: 所有 `[barcode]` 用例 PASS（QR 解码文本精确匹配）。

- [ ] **Step 4: Commit**

```bash
git add tests/test_barcode.cpp tests/CMakeLists.txt
git commit -m "test(barcode): unit tests for BarcodeDetector QR decode"
```

---

### Task 6: Python 绑定 barcode_pybind.cpp

**Files:**
- Create: `csrc/pybind/vision/barcode_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（forward decl 行 ~38 + bind_vision 调用行 ~73）

**Interfaces:**
- Consumes: `BarcodeDetector`、`BarcodeResult`、`Formats` 常量（Task 3/4）。
- Produces: pybind 类 `modeldeploy.vision.BarcodeDetector`（`set_formats` / `detect` / `formats`）+ `BarcodeResult` 值类 + `FMT_*` 常量。

- [ ] **Step 1: 写 barcode_pybind.cpp**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vision/barcode/barcode.h"
#include "vision/barcode/result.h"

namespace modeldeploy::vision {
    void bind_barcode(const pybind11::module& m) {
        using namespace barcode;
        m.attr("FMT_QR_CODE") = FMT_QR_CODE;
        m.attr("FMT_DATA_MATRIX") = FMT_DATA_MATRIX;
        m.attr("FMT_AZTEC") = FMT_AZTEC;
        m.attr("FMT_EAN_8") = FMT_EAN_8;
        m.attr("FMT_EAN_13") = FMT_EAN_13;
        m.attr("FMT_UPC_A") = FMT_UPC_A;
        m.attr("FMT_UPC_E") = FMT_UPC_E;
        m.attr("FMT_CODE_128") = FMT_CODE_128;
        m.attr("FMT_CODE_39") = FMT_CODE_39;
        m.attr("FMT_CODE_93") = FMT_CODE_93;
        m.attr("FMT_ITF") = FMT_ITF;
        m.attr("FMT_CODABAR") = FMT_CODABAR;
        m.attr("FMT_ALL") = FMT_ALL;

        pybind11::class_<BarcodeResult>(m, "BarcodeResult")
            .def(pybind11::init<>())
            .def_readwrite("text", &BarcodeResult::text)
            .def_readwrite("format", &BarcodeResult::format)
            .def_readwrite("quad", &BarcodeResult::quad)
            .def_readwrite("score", &BarcodeResult::score)
            .def_readwrite("is_qr", &BarcodeResult::is_qr)
            .def("__repr__", [](const BarcodeResult& r) {
                return "BarcodeResult(text=" + r.text + ", format=" + r.format
                       + ", is_qr=" + std::to_string(r.is_qr) + ")";
            });

        pybind11::class_<BarcodeDetector>(m, "BarcodeDetector")
            .def(pybind11::init<>())
            .def("set_formats", &BarcodeDetector::set_formats,
                 pybind11::arg("formats"))
            .def("detect", &BarcodeDetector::detect, pybind11::arg("img"))
            .def_property_readonly("formats", &BarcodeDetector::formats);
    }
}
```

- [ ] **Step 2: vision_pybind.cpp 注册**

在 `bind_tracking` 前导声明附近（行 ~38）加入：
```cpp
    void bind_barcode(const pybind11::module&);
```
在 `bind_vision()` 末尾（`bind_tracking(m);` 之后，行 ~73）加入：
```cpp
    bind_barcode(m);
```

- [ ] **Step 3: Python smoke 测试**

构建 Python 模块（`build_py`），运行：
```bash
python -c "import modeldeploy.vision as v; d=v.BarcodeDetector(); print(d.formats()); d.set_formats(v.FMT_QR_CODE); print('ok', d.__class__.__name__)"
```
Expected: 能构造 `BarcodeDetector`、读取/设置 formats、无导入错误。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/vision/barcode_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(pybind): barcode BarcodeDetector bindings"
```

---

### Task 7: CAPI md_barcode_*

**Files:**
- Modify: `capi/md_capi.h`
- Modify: `capi/md_capi.cpp`

**Interfaces:**
- Consumes: `BarcodeDetector`、`BarcodeResult`、`ImageData`、`MDImageHandle`（capi 图像句柄）。
- Produces: CAPI 函数（供 C#/Rust/外部 C 使用）：
  ```c
  typedef struct md_barcode_handle* MDBarcodeHandle;
  MD_CAPI_EXPORT MDStatus md_barcode_create(MDBarcodeHandle* out);
  MD_CAPI_EXPORT void md_barcode_destroy(MDBarcodeHandle h);
  MD_CAPI_EXPORT MDStatus md_barcode_set_formats(MDBarcodeHandle h, uint32_t formats);
  MD_CAPI_EXPORT MDStatus md_barcode_detect(MDBarcodeHandle h, MDImageHandle img,
                                            MD_BarcodeItem* out_items, uint32_t* out_count);
  ```
  其中：
  ```c
  typedef struct MD_BarcodeItem {
      float quad[8];      // 左上前右下... 4 点 (x0,y0,x1,y1,x2,y2,x3,y3)
      float score;
      int32_t is_qr;
      char format[16];    // UTF-8，如 "QR_CODE"
  } MD_BarcodeItem;
  ```
- 内存约定：`out_items` 由调用方提供（数组容量 >= *out_count），`md_barcode_detect` 写入并置 `*out_count` 为实际写入数；`format` 为固定 `char[16]`，文本两端点用 `md_barcode_detect_text`（可选）或主 detect 用固定长度拷贝。**为简化 C#/Rust 互操作，文本采用单独 API 传入缓冲区**：
  ```c
  MD_CAPI_EXPORT MDStatus md_barcode_detect_text(MDBarcodeHandle h, int32_t index,
                                                 MDImageHandle img, char* text, uint32_t text_cap);
  ```
  （index 为 detect 后结果索引；也可改为 detect 直接给固定 256B 文本缓冲。实现时二选一并保持 C#/Rust 一致——推荐 detect 内带 `char text[256]` 固定字段，最简单互操作。）

> **互操作简化决策**：为让 C#/Rust 好处理变长字符串，**最终以 md_capi.h 实际定义为准**：`MD_BarcodeItem` 含固定 `char text[256]` + `char format[16]` + `float quad[8]` + `float score` + `int32_t is_qr`，`md_barcode_detect(h, img, MD_BarcodeItem* items, uint32_t* count)` 单次调用即可。请固定该结构，C#/Rust 严格镜像。

- [ ] **Step 1: md_capi.h 新增类型与函数声明**

参照 tracking 的 `MDTrackItem`/`md_tracker_*` 区块（行 465-522）新增：
```c
typedef struct MD_BarcodeItem {
    char text[256];      // 解码文本 UTF-8
    char format[16];     // 格式，如 "QR_CODE"
    float quad[8];       // 4 点坐标 (x0,y0,x1,y1,x2,y2,x3,y3)
    float score;
    int32_t is_qr;
} MD_BarcodeItem;
typedef struct md_barcode_handle* MDBarcodeHandle;

MD_CAPI_EXPORT MDStatus md_barcode_create(MDBarcodeHandle* out);
MD_CAPI_EXPORT void md_barcode_destroy(MDBarcodeHandle h);
MD_CAPI_EXPORT MDStatus md_barcode_set_formats(MDBarcodeHandle h, uint32_t formats);
MD_CAPI_EXPORT MDStatus md_barcode_detect(MDBarcodeHandle h, MDImageHandle img,
                                          MD_BarcodeItem* items, uint32_t* count);
```

- [ ] **Step 2: md_capi.cpp 实现**

参照 `md_tracker_handle`（`md_capi.cpp:108-112`、`md_tracker_create/destroy` 3314-3335）：
```cpp
struct md_barcode_handle {
    modeldeploy::vision::barcode::BarcodeDetector det;
    uint32_t formats = modeldeploy::vision::barcode::FMT_ALL;
};

MDStatus md_barcode_create(MDBarcodeHandle* out) {
    if (!out) { set_error("md_barcode_create: out is null"); return MD_ERR_NULL_POINTER; }
    auto* h = new md_barcode_handle();
    *out = h;
    return MD_OK;
}
void md_barcode_destroy(MDBarcodeHandle h) { delete static_cast<md_barcode_handle*>(h); }

MDStatus md_barcode_set_formats(MDBarcodeHandle h, uint32_t formats) {
    if (!h) { set_error("md_barcode_set_formats: h null"); return MD_ERR_NULL_POINTER; }
    auto* bh = static_cast<md_barcode_handle*>(h);
    bh->formats = formats;
    bh->det.set_formats(formats);
    return MD_OK;
}

MDStatus md_barcode_detect(MDBarcodeHandle h, MDImageHandle img,
                           MD_BarcodeItem* items, uint32_t* count) {
    if (!h || !img || !count) {
        set_error("md_barcode_detect: null arg"); return MD_ERR_NULL_POINTER;
    }
    const auto& image = get_image_data(img);   // 从 MDImageHandle 取 ImageData；以现有实现为准
    auto res = static_cast<md_barcode_handle*>(h)->det.detect(image);
    uint32_t need = static_cast<uint32_t>(res.size());
    if (items == nullptr) { *count = need; return MD_OK; }  // 容量查询
    uint32_t cap = *count;
    uint32_t n = std::min(cap, need);
    for (uint32_t i = 0; i < n; ++i) {
        auto& r = res[i];
        MD_BarcodeItem& it = items[i];
        memset(&it, 0, sizeof(it));
        memcpy(it.text, r.text.c_str(), std::min<size_t>(r.text.size(), 255));
        memcpy(it.format, r.format.c_str(), std::min<size_t>(r.format.size(), 15));
        for (int k = 0; k < 4; ++k) {
            it.quad[2 * k] = r.quad[k].x;
            it.quad[2 * k + 1] = r.quad[k].y;
        }
        it.score = r.score;
        it.is_qr = r.is_qr ? 1 : 0;
    }
    *count = n;
    return MD_OK;
}
```
> **实现注意**：`get_image_data(img)` 取 `ImageData` 的实际函数需在 `md_capi.cpp` 中核对其现有实现（IMAGE 相关函数内），确保用对接口。若超容量返回 `MD_ERR_BUFFER_TOO_SMALL` 也需新建该枚举或复用 `MD_ERR_INVALID_ARGUMENT`——按现有枚举复用即可（建议返回 `MD_ERR_INVALID_ARGUMENT`）。

- [ ] **Step 3: CAPI 测试**

在 `tests/test_capi.cpp` 增加 `md_barcode_*` 契约测试（创建/销毁/null 守卫/容量查询/detect）。用 QR 样本 `MDImageHandle`（`md_image_from_file` 读测试图，或内存生成后 `md_image_from_*24` 构造）：
- 断言 create 返回 `MD_OK` 且 handle 非空；
- 断言 null 传递返回 `MD_ERR_NULL_POINTER`；
- 断言 `items==nullptr` 时 `*count` 返回需要数、不崩溃；
- 断言 detect 能解码已知 QR 并 `text` 匹配。

- [ ] **Step 4: 构建 + 运行 `[capi]`**

用 .bat 包裹 vcvars64 `cmake --build build_tdc_gpu`，然后 `test_modeldeploy.exe "[capi]"`。Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): md_barcode_create/destroy/set_formats/detect"
```

---

### Task 8: C# 绑定 BarcodeDetector

**Files:**
- Create: `csharp/ModelDeploy/BarcodeDetector.cs`
- Modify: `csharp/ModelDeploy/NativeMethods.cs`（+ `#region 条码识别`）
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（+ `MDBarcodeItem` + `MDBarcodeHandle`）
- Create: `csharp/ModelDeployUnitTest/BarcodeTests.cs`

**Interfaces:**
- Consumes: `API`（Task 7）、`VisionImage`（csharp 图像封装，`VisionImage.cs`，`Handle` 属性）。
- Produces: `public sealed class BarcodeDetector : IDisposable`（`Detect(VisionImage)` 返回 `BarcodeResult[]`、`SetFormats(uint)`）、NUnit 单测。

- [ ] **Step 1: types_internal_c.cs 新增类型**

（仿现 `MDTrackItem` 定义）加入：
```csharp
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
internal struct MDBarcodeItem
{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string Text;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 16)] public string Format;
    public float Quad0_x; public float Quad0_y;
    public float Quad1_x; public float Quad1_y;
    public float Quad2_x; public float Quad2_y;
    public float Quad3_x; public float Quad3_y;
    public float Score;
    public int IsQr;
}
```

- [ ] **Step 2: NativeMethods.cs 新增 region**

仿 tracking region（行 393-420）：
```csharp
#region 条码识别
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_barcode_create(out IntPtr handle);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void md_barcode_destroy(IntPtr handle);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_barcode_set_formats(IntPtr handle, uint formats);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_barcode_detect(IntPtr handle, IntPtr image,
    [Out] MDBarcodeItem[] items, ref uint count);
#endregion
```

- [ ] **Step 3: 新建 BarcodeDetector.cs**

仿 `Tracker.cs`（sealed + IDisposable + `_handle` + `BaseModel.GetLastError()`）：
```csharp
namespace ModelDeploy
{
    public sealed class BarcodeResult
    {
        public string Text { get; set; }
        public string Format { get; set; }
        public PointF[] Quad { get; set; }
        public float Score { get; set; }
        public bool IsQr { get; set; }
    }

    public sealed class BarcodeDetector : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public BarcodeDetector()
        {
            if (NativeMethods.md_barcode_create(out _handle) != MDStatus.MD_OK)
                throw new InvalidOperationException($"barcode create failed: {BaseModel.GetLastError()}");
        }

        public void SetFormats(uint formats)
        {
            EnsureNotDisposed();
            if (NativeMethods.md_barcode_set_formats(_handle, formats) != MDStatus.MD_OK)
                throw new InvalidOperationException($"set_formats failed: {BaseModel.GetLastError()}");
        }

        public BarcodeResult[] Detect(VisionImage image)
        {
            EnsureNotDisposed();
            uint need = 0;
            NativeMethods.md_barcode_detect(_handle, image.Handle, null, ref need);
            if (need == 0) return Array.Empty<BarcodeResult>();
            var items = new MDBarcodeItem[need];
            uint written = need;
            if (NativeMethods.md_barcode_detect(_handle, image.Handle, items, ref written) != MDStatus.MD_OK)
                throw new InvalidOperationException($"detect failed: {BaseModel.GetLastError()}");
            var result = new BarcodeResult[written];
            for (int i = 0; i < (int)written; i++)
                result[i] = new BarcodeResult {
                    Text = items[i].Text, Format = items[i].Format,
                    Quad = new[] { new PointF(items[i].Quad0_x, items[i].Quad0_y),
                                   new PointF(items[i].Quad1_x, items[i].Quad1_y),
                                   new PointF(items[i].Quad2_x, items[i].Quad2_y),
                                   new PointF(items[i].Quad3_x, items[i].Quad3_y) },
                    Score = items[i].Score, IsQr = items[i].IsQr != 0 };
            return result;
        }

        private void EnsureNotDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BarcodeDetector));
        }

        public void Dispose()
        {
            if (_disposed) return;
            if (_handle != IntPtr.Zero) NativeMethods.md_barcode_destroy(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~BarcodeDetector() { Dispose(); }
    }
}
```
> 核对：`PointF` 若不存在于命名空间，改用自定义 `QuadPoint` 结构（x,y float）。`VisionImage.Handle` 属性名以 `VisionImage.cs` 实际为准核对。

- [ ] **Step 4: 新建 BarcodeTests.cs**

仿 `TrackerTests.cs`：生成/读取 QR 样本图（`VisionImage.Read(path)` 或内存），断言 `Detect` 返回非空且首项 `IsQr`、`Text` 匹配：
```csharp
[Test]
public void BarcodeDetect_QR_ReturnsText()
{
    using var det = new BarcodeDetector();
    // 使用 test_data 下 QR 样本（或用内存生成）
    using var img = VisionImage.Read("../../test_data/qr_sample.png");
    var res = det.Detect(img);
    Assert.That(res, Is.Not.Empty);
    Assert.That(res[0].IsQr, Is.True);
}
```
> 需在 `test_data/` 放一个已知 QR 样本 `qr_sample.png`（若 C++ 测试用内存生成，C# 测试可同样用 OpenCV 生成器；否则共用该样本图，保证跨语言同一断言文本）。

- [ ] **Step 5: 构建 + 运行 dotnet test**

```bash
dotnet build csharp/csharp.sln
dotnet test csharp/ModelDeployUnitTest/ModelDeployUnitTest.csproj
```
Expected: 0 构建错误 + BarcodeTests 通过（需 build_tdc_gpu 的 `ModelDeploySDK.dll` 可被找到，`ModelDeploySdkDir` 指向 build_tdc_gpu）。

- [ ] **Step 6: Commit**

```bash
git add csharp/ModelDeploy/BarcodeDetector.cs csharp/ModelDeploy/NativeMethods.cs csharp/ModelDeploy/types_internal_c.cs csharp/ModelDeployUnitTest/BarcodeTests.cs
git commit -m "feat(csharp): BarcodeDetector binding + tests"
```

---

### Task 9: Rust 绑定 BarcodeDetector

**Files:**
- Create: `rust/modeldeploy/src/barcode.rs`
- Modify: `rust/modeldeploy/src/ffi.rs`（+ `MDBarcodeHandle` + `md_barcode_*` + `MDBarcodeItem`）
- Modify: `rust/modeldeploy/src/types.rs`（+ `BarcodeResult`）
- Modify: `rust/modeldeploy/src/lib.rs`（+ `pub mod barcode;` + re-export）
- Modify: `rust/modeldeploy/tests/integration_test.rs`（+ `test_barcode_qr_decode`）

**Interfaces:**
- Consumes: API（Task 7）、`ffi` 类型、`crate::error::check_status`。
- Produces: `BarcodeDetector`（`new()/set_formats()/detect(&Image)->Result<Vec<BarcodeResult>>`）、`BarcodeResult`、`BarcodeFormat`。

- [ ] **Step 1: ffi.rs 扩展**

```rust
pub type MDBarcodeHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MDBarcodeItem {
    pub text: [c_char; 256],
    pub format: [c_char; 16],
    pub quad: [f32; 8],
    pub score: f32,
    pub is_qr: i32,
}
// extern 块内：
pub fn md_barcode_create(out: *mut MDBarcodeHandle) -> MDStatus;
pub fn md_barcode_destroy(h: MDBarcodeHandle);
pub fn md_barcode_set_formats(h: MDBarcodeHandle, formats: c_uint) -> MDStatus;
pub fn md_barcode_detect(h: MDBarcodeHandle, img: MDImageHandle,
                         items: *mut MDBarcodeItem, count: *mut c_uint) -> MDStatus;
```

- [ ] **Step 2: types.rs 新增**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum BarcodeFormat {
    QrCode, DataMatrix, Aztec, Ean8, Ean13, Upca, Upce,
    Code128, Code39, Code93, Itf, Codabar, Other(String),
}
impl BarcodeFormat { fn from_name(s: &str) -> Self { ... } }

#[derive(Debug, Clone)]
pub struct BarcodeResult {
    pub text: String,
    pub format: BarcodeFormat,
    pub quad: [(f32, f32); 4],
    pub score: f32,
    pub is_qr: bool,
}
```

- [ ] **Step 3: 新建 barcode.rs**

仿 `tracker.rs`：
```rust
use crate::ffi::{self, MDBarcodeHandle};
use crate::error::{check_status, MdError};
use crate::image::Image;
use crate::types::BarcodeResult;
use std::ffi::CString;
use std::ptr;

pub struct BarcodeDetector { handle: MDBarcodeHandle }
unsafe impl Send for BarcodeDetector {}
unsafe impl Sync for BarcodeDetector {}
impl BarcodeDetector {
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_barcode_create(&mut h) })?;
        if h.is_null() { return Err(MdError::ModelInit("barcode".into())); }
        Ok(Self { handle: h })
    }
    pub fn set_formats(&self, formats: u32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_barcode_set_formats(self.handle, formats) })
    }
    pub fn detect(&self, img: &Image) -> Result<Vec<BarcodeResult>, MdError> {
        let mut need: u32 = 0;
        check_status(unsafe { ffi::md_barcode_detect(self.handle, img.handle, ptr::null_mut(), &mut need) })?;
        let mut items = vec![ffi::MDBarcodeItem::default(); need as usize];
        let mut written = need;
        check_status(unsafe { ffi::md_barcode_detect(self.handle, img.handle, items.as_mut_ptr(), &mut written) })?;
        let mut out = Vec::with_capacity(written as usize);
        for i in 0..written as usize {
            let it = items[i];
            let text = cstr_to_string(&it.text);
            let format = cstr_to_string(&it.format);
            out.push(BarcodeResult {
                text,
                format: BarcodeFormat::from_name(&format),
                quad: [ (it.quad[0], it.quad[1]),(it.quad[2], it.quad[3]),
                        (it.quad[4], it.quad[5]),(it.quad[6], it.quad[7]) ],
                score: it.score,
                is_qr: it.is_qr != 0,
            });
        }
        Ok(out)
    }
}
impl Drop for BarcodeDetector {
    fn drop(&mut self) {
        if !self.handle.is_null() { unsafe { ffi::md_barcode_destroy(self.handle) }; self.handle = ptr::null_mut(); }
    }
}
```
> 需 `MDBarcodeItem::default()`（为 `[c_char;N]` derive Default）与 `cstr_to_string` 工具（可放 ffi.rs 或 barcode.rs，仿 types.rs 中 OcrLine.text 的解析）。`Image` 的 `handle` 字段名以 `image.rs` 实际为准。

- [ ] **Step 4: lib.rs 注册**

```rust
pub mod barcode;
pub use barcode::BarcodeDetector;
```

- [ ] **Step 5: integration_test.rs 新增用例**

仿 `test_tracker_byte_track_stable_id`（行 523-547）：
```rust
#[test]
fn test_barcode_qr_decode() -> Result<()> {
    let det = BarcodeDetector::new()?;
    let img = Image::read(&test_img("qr_sample.png"))?;   // 依赖 test_data/qr_sample.png
    let res = det.detect(&img)?;
    assert!(!res.is_empty());
    assert!(res[0].is_qr);
    assert_eq!(res[0].text, "https://example.com/MD");
    Ok(())
}
```

- [ ] **Step 6: cargo build + test**

```bash
cd rust/modeldeploy
$env:MODELDEPLOY_LIB_DIR="E:\CLionProjects\ModelDeploy\build_tdc_gpu\bin"
cargo build
cargo test --test integration_test test_barcode_qr_decode
```
Expected: 编译 0 错误 + 用例 PASS（需 `test_data/qr_sample.png` 存在）。

- [ ] **Step 7: Commit**

```bash
git add rust/modeldeploy/src/barcode.rs rust/modeldeploy/src/ffi.rs rust/modeldeploy/src/types.rs rust/modeldeploy/src/lib.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(rust): BarcodeDetector binding + test"
```

---

### Task 10: Demo + 文档

**Files:**
- Create: `examples/demo_barcode/demo_barcode.cpp`
- Create: `examples/demo_barcode/CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`（加 `add_subdirectory(demo_barcode)`）

**Interfaces:**
- Consumes: `BarcodeDetector`、`ImageData`/OpenCV Mat。
- Produces: 可运行 `demo_barcode` 可执行文件（读图 → 打印解码结果）。

- [ ] **Step 1: 写 demo_barcode.cpp**

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include "vision/common/image_data.h"
#include "vision/barcode/barcode.h"

// 将 cv::Mat 转 ImageData 的辅助（按 image_data.h 签名）
static modeldeploy::vision::ImageData mat_to_image(const cv::Mat& img) {
    modeldeploy::vision::ImageData im;
    im.width = img.cols; im.height = img.rows; im.channels = img.channels();
    im.data = img.data;
    return im;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: demo_barcode <image>\n"; return 1; }
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) { std::cerr << "failed to read " << argv[1] << "\n"; return 1; }
    auto image = mat_to_image(img);
    modeldeploy::vision::barcode::BarcodeDetector det;
    auto res = det.detect(image);
    if (res.empty()) { std::cout << "no barcode found\n"; return 0; }
    for (auto& r : res) {
        std::cout << "[" << r.format << "] " << r.text
                  << " (score=" << r.score << ", is_qr=" << r.is_qr << ")\n";
    }
    return 0;
}
```

- [ ] **Step 2: 写 demo_barcode/CMakeLists.txt**

```cmake
# 纯 CV 演示，无需后端变体
add_executable(demo_barcode demo_barcode.cpp)
target_link_libraries(demo_barcode PUBLIC ${LIBRARY_NAME})
```

- [ ] **Step 3: examples/CMakeLists.txt 注册**

在 `add_subdirectory(demo_tracking)`（行 ~53）附近新增：
```cmake
add_subdirectory(demo_barcode)
```

- [ ] **Step 4: 构建并运行 demo**

用 .bat 包裹 vcvars64 `cmake --build build_tdc_gpu`，然后：
```bat
build_tdc_gpu\examples\demo_barcode\demo_barcode.exe test_data\qr_sample.png
```
Expected: 打印 `[QR_CODE] https://example.com/MD (score=1, is_qr=1)`。

- [ ] **Step 5: 更新文档**

在 `examples/EXAMPLES.md` 追加 demo_barcode 说明；必要时在 README 功能矩阵补"条码/二维码"。（若 README 有视觉能力清单，补一行 `Barcode/QR`。）

- [ ] **Step 6: Commit**

```bash
git add examples/demo_barcode examples/CMakeLists.txt examples/EXAMPLES.md
git commit -m "feat(examples): demo_barcode + docs"
```

---

### Task 11: 全量构建 + 全后端语义确认 + 收尾

**Files:**
- 无新增；验证整个分支。

**Interfaces:**
- 验证所有 Task 1-10 的产物协同工作。

- [ ] **Step 1: 全量 C++ 测试**

用 .bat 包裹 vcvars64 `cmake --build build_tdc_gpu`，然后：
```bat
cd build_tdc_gpu
test_modeldeploy.exe "[barcode]"
test_modeldeploy.exe "[capi]"
test_modeldeploy.exe "[tracking]"   # 确认未回归
test_modeldeploy.exe "[core]"
```
Expected: 全部 PASS，tracking/core 无回归。

- [ ] **Step 2: 全套绑定复验**

- Python：`python -c "import modeldeploy; from modeldeploy.vision import BarcodeDetector`（构建 `build_py` 后）。
- C#：`dotnet test csharp/ModelDeployUnitTest`。
- Rust：`cargo test`（MODELDEPLOY_LIB_DIR 指向 build_tdc_gpu/bin）。

- [ ] **Step 3: 跨后端语义断言（静态）**

确认 `BarcodeDetector` 不引用任何 ORT/MNN/TRT/SOPHGO API（grep `BarcodeDetector` 相关源文件无 backend include）。纯 CV → 天然跨所有后端。在报告/文档中如实声明"各后端语义一致，无 DNN 推理"。

- [ ] **Step 4: 复核设计"明确不做"项**

确认未引入 DNN 定位增强、未做视频扫描、未做解码失败仅检测框。（与设计一致。）

- [ ] **Step 5: 提交收尾commit（如有文档变更）**

```bash
git add -A
git commit -m "docs(barcode): finalize feature"   # 如无变更则跳过
```
（不必要时不要制造空 commit。）

---

## Self-Review Checklist

- **Spec coverage**：设计中的 架构（BarcodeDetector/Reader/result）、结果结构字段（text/format/quad/score/is_qr）、`set_formats`/`detect` API、全套绑定（C++/py/CAPI/C#/Rust）、demo、测试、明确的"不做"项 —— 已在 Task 2-11 覆盖。`third_party/zxing-cpp` 捆绑与 `BUILD_BARCODE` 选项 —— Task 1。测试数据/样本分散在 Task 5/8/9（QR 样本图 `test_data/qr_sample.png` 或内存生成）。
- **Placeholder 扫描**：无 TBD/TODO；`实现注意` 均给出具体核对点与替代方案。
- **Type/命名一致性**：`BarcodeResult`（text/format/quad/score/is_qr）贯穿 Task 2-11；`Formats` 与 `FMT_*` 常量在 Task 3 定义、Task 6 pybind 复用、Task 7 CAPI `uint32_t`、Task 8/9 镜像。`BarcodeDetector`/`set_formats`/`detect` 签名在 C++/py/CAPI/C#/Rust 保持一致。CAPI `MD_BarcodeItem` 字段（text[256]/format[16]/quad[8]/score/is_qr）在 C#/Rust 严格镜像（Task 7/8/9）。

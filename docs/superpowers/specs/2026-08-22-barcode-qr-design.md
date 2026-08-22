# Barcode / QR 条码识别模块设计

日期：2026-08-22
状态：已确认（方案 A：ZXing-C++）
分支：main（功能在 main 上以 feature 分支开发后合并）

## 背景与目标

在 ModelDeploy SDK 中新增条码 / 二维码识别能力。该能力属于**纯经典 CV（非 DNN）**，
因此**天然跨全部推理后端（ORT / MNN / TRT / SOPHGO），零 DNN 推理依赖**。

目标：输入单张图片（`cv::Mat` / RGB 或 BGR），输出图内所有可解码码的
文本、格式、定位四边形、可靠度。交付形态与已合入的 tracking 家族对齐：
C++ + Python + CAPI + C# + Rust + demo + 测试（全套绑定）。

## 关键决策

- **解码库：ZXing-C++（捆绑 `third_party/zxing-cpp`）**。
  - 原因：捆绑的 OpenCV 5（Windows 预编译）**无 objdetect 模块**（无 QRCodeDetector、无
    barcode），无法走 OpenCV；ZXing-C++ 是开源条码/二维码解码的事实标准，成熟、无外部依赖、
    纯 C++ 可离线编译、跨全后端。
  - 支持格式：QR / DataMatrix / Aztec / EAN-8/13 / UPC-A/E / Code128 / Code39 / Code93 /
    ITF / Codabar 等。
- **一体式检测+解码**：ZXing 内部先定位（finder pattern / 条码边界，产出 quad）再解码。
  一次调用同时给出 `quad` + `text`。无独立的"只检测、不支持解码"模式（已记录为已知限制，
  DNN 定位增强列为后续可选，不在首版范围）。

## 架构

```
csrc/vision/barcode/
├── barcode.h / barcode.cpp      # BarcodeDetector 外观类（公共 API）
├── result.h                     # BarcodeResult 结果结构
├── barcode_reader.h/.cpp        # ZXing 读码封装（多格式解码 + 坐标映射）
└── preprocess.h/.cpp            # 灰度/ROI 预处理（区域提取）
```

- `BarcodeDetector` 依赖 ZXing-C++（编译期依赖），输入图像输出 `std::vector<BarcodeResult>`。
- 支持单图多码（遍历 ZXing results）。
- 坐标映射：ZXing 结果坐标 → 归一化/原图坐标的 `quad`。
- 预处理：灰度转换（若输入彩色）、可选 ROI 提取。

## 结果结构

```cpp
namespace modeldeploy::vision {
    struct MODELDEPLOY_CXX_EXPORT BarcodeResult {
        std::string text;              // 解码文本 / URL
        std::string format;            // "QR_CODE"/"EAN_13"/"CODE_128"/"DATA_MATRIX"/...
        std::array<Point2f, 4> quad;   // 码区域四角定位框（原图坐标，左上起顺时针）
        float score;                   // 可靠度 [0,1]
        bool is_qr;                    // 是否二维码
    };
}
```

`Point2f`、`Rect2f` 复用 `vision/common/struct.h` 现有定义，不重复定义。

## 数据流

```
输入图像(cv::Mat)
  → preprocess（灰度 / 可选 ROI）
  → BarcodeReader::read 遍历 ZXing 结果
  → 填充 BarcodeResult（text / format / quad / score / is_qr）
  → 返回 std::vector<BarcodeResult>
```

## API 设计（外观类）

```cpp
class MODELDEPLOY_CXX_EXPORT BarcodeDetector {
 public:
  BarcodeDetector();                       // 默认：启用常用格式
  void set_formats(const std::vector<std::string>& formats); // 限定格式子集
  std::vector<BarcodeResult> detect(const cv::Mat& img) const; // 同步检测+解码
  // 若输入非灰度且为 RGB，内部转 BGR/灰度再送入读码器
};
```

## 绑定（全套，对齐 tracking）

- **Python**：`BarcodeDetector`（`__init__`、`set_formats`、`detect`），自定义返回值绑定为
  `List[BarcodeResult]`，`BarcodeResult` 含 `text/format/quad/score/is_qr` 字段 + `__repr__`。
  在 `vision_pybind.cpp` 注册 `bind_barcode`。
- **CAPI**（`capi/md_capi.h/.cpp`）：
  - `md_barcode_create()` / `md_barcode_destroy()`
  - `md_barcode_set_formats()`
  - `md_barcode_detect()`：输入 `MDImage`，输出 `MD_BarcodeResult` 数组（count + 每项含
    text/format/quad/score/is_qr）。沿用现有 `MDImage`/`MDBox` 等 MD 结构与内存约定。
- **C#**：`BarcodeDetector` 封装（对齐 `Tracker.cs` 风格），`NativeMethods.cs` 增加对应
  P/Invoke，`ModelDeployUnitTest` 增加单测（QR 合成样本）。
- **Rust**：`barcode.rs` + `ffi.rs` 扩展 + `types.rs` 增加 `BarcodeResult`，`integration_test`
  增加 QR 解码用例。
- **demo**：`examples/demo_barcode/`，沿用 `md_add_demo_matrix(STEM ...)` 生成各后端变体；
  由于纯 CV 无 DNN，各变体共用同一实现（后端无关），演示读图 → 打印/绘制定位框。

## 测试

- **单元**：OpenCV 生成二维码样本（`cv::qr::generate` 或捆绑样本）与已知 1D 条码样本，
  断言：解码文本正确、format 正确、quad 四点合理、is_qr 正确、多码图返回多条。
- **CAPI**：`md_barcode_*` 契约（创建/销毁/格式限定/null 守卫/返回数组与 count）。
- **Rust**：`cargo test`，QR 解码 + 坐标映射。
- **C#**：`dotnet test`，QR 解码。
- 测试数据：合成/内置生成，必要时外链权重/样本（本模块无模型权重，仅需样本图）。

## 明确不做（YAGNI / 后续可选）

- 不引入 DNN 定位模型增强（严重模糊/超小 1D 码的极限场景留待后续）。
- 不做视频流连续扫描 / 跟踪式扫码（属 pipeline 层，不在本模块）。
- 不做解码失败时"仅返回检测框"的独立接口（ZXing 一体式已知限制，后续可选）。

## 已知限制（诚实声明）

- 严重模糊 / 超低分辨率 / 重度磨损的码，解码可能失败（无法返回框）。ZXing-C++ 对清晰/中等
  质量码识别率高，是开源最优解；极限场景弱于商业 SDK（Dynamsoft / Scandit），属后续增强范畴。

## 构建影响

- `third_party/` 新增 zxing-cpp（源码捆绑或 FetchContent，模型平台归档）。
- 需在顶部 CMake 将 zxing-cpp 加入 `VISION_SOURCE`/链接；确保各后端（ORT/MNN/TRT/SOPHGO）
  都能编译（该模块不依赖任何推理后端库）。
- `tests/CMakeLists.txt` 显式 TEST_SOURCES 注册 `test_barcode.cpp`。
- 新增 `examples/CMakeLists.txt` 的 demo_barcode 矩阵条目。

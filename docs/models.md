# ModelDeploy 模型详解

按功能点介绍 ModelDeploy 支持的全部模型：接口、输入输出、示例。所有模型类均继承 `BaseModel`，提供统一的 `predict` / `batch_predict` / `clone` 接口。

## 1. 目标检测（Detection）

### 1.1 UltralyticsDet（YOLO 系列）

通用目标检测，支持 YOLOv5/v8/v9/v11/v12 等 Ultralytics 系列。

```cpp
#include "modeldeploy/vision.h"

modeldeploy::RuntimeOption option;
option.use_ort_backend(); option.use_cpu();

auto det = modeldeploy::vision::detection::UltralyticsDet("yolo11n.onnx", option);
det.get_preprocessor().set_size({640, 640});
det.get_postprocessor().set_conf_threshold(0.25f);
det.get_postprocessor().set_nms_threshold(0.5f);

auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::DetectionResult> result;
det.predict(img, &result);
// result[i]: {box(x,y,w,h), label_id, score}
```

**关键接口**：

| 接口 | 说明 |
|------|------|
| `batch_predict(images, results)` | 批量推理 |
| `clone()` | 多线程克隆 |
| `predict_nv12(y, uv, w, h, ...)` | NV12 直通（GPU 场景） |
| `get_preprocessor().set_size({640,640})` | 输入尺寸 |
| `get_postprocessor().set_conf_threshold()` | 置信度阈值 |
| `get_label_map("names")` | 类别名文件 |

**示例**：`examples/demo_det/`
- `demo_detection_cxx.cpp` — 基础用法
- `demo_detection_batch.cpp` — 批量推理
- `demo_detection_multi_thread.cpp` — 多线程
- `demo_detection_sophgo.cpp` — Sophgo TPU

## 2. 实例分割（Instance Segmentation）

### 2.1 UltralyticsSeg

```cpp
auto seg = modeldeploy::vision::detection::UltralyticsSeg("yolo11n-seg.onnx", option);
seg.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::InstanceSegResult> result;
seg.predict(img, &result);
// result[i]: {box, mask(二值图), label_id, score}
```

mask 以 `Mask` 结构保存（shape `{h, w}`，uint8 0/1），可用 `vis_iseg` 可视化。

**示例**：`examples/demo_iseg/demo_instance_seg_cxx.cpp`

## 3. 姿态估计（Pose / Keypoints）

### 3.1 UltralyticsPose

```cpp
auto pose = modeldeploy::vision::detection::UltralyticsPose("yolo11n-pose.onnx", option);
pose.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::KeyPointsResult> result;
pose.predict(img, &result);
// result[i]: {box, keypoints[N][3](x,y,conf), score}
```

**示例**：`examples/demo_kps/demo_pose_cxx.cpp`

## 4. 旋转框检测（Oriented Bounding Box）

### 4.1 UltralyticsObb

```cpp
auto obb = modeldeploy::vision::detection::UltralyticsObb("yolo11n-obb.onnx", option);
obb.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::ObbResult> result;
obb.predict(img, &result);
// result[i]: {rotated_box(四点), label_id, score}
```

**示例**：`examples/demo_obb/demo_obb_cxx.cpp`

## 5. 图像分类（Classification）

### 5.1 Classification

```cpp
auto cls = modeldeploy::vision::classification::Classification("yolo11n-cls.onnx", option);
cls.get_preprocessor().set_size({224, 224});

std::vector<modeldeploy::vision::ClassifyResult> result;
cls.predict(img, &result);
// result[i]: {label_ids, scores}
```

支持 `set_topk`、`set_multi_label`（多标签分类）。

**示例**：`examples/demo_cls/demo_classification_cxx.cpp`

## 6. 人脸（Face）

人脸模块包含完整人脸应用链：检测、识别、年龄、性别、防伪。

| 模型类 | 功能 | 输出 |
|--------|------|------|
| `Scrfd` | 人脸检测 | 框 + 5 关键点 |
| `SeetaFaceID` | 人脸特征提取 | 512 维 embedding |
| `SeetaFaceAge` | 年龄估计 | 年龄段 |
| `SeetaFaceGender` | 性别判定 | 男/女 |
| `SeetaFaceAsFirst` | 防伪一阶段 | 分类结果 |
| `SeetaFaceAsSecond` | 防伪二阶段 | 模糊/活体/翻拍概率 |
| `SeetaFaceAsPipeline` | 防伪流水线 | 一二阶段串联 |
| `FaceRecognizerPipeline` | 识别流水线 | 检测 + 特征一体化 |

```cpp
// 人脸检测
auto det = modeldeploy::vision::face::Scrfd("scrfd.onnx", option);
det.get_preprocessor().set_size({640, 640});

std::vector<modeldeploy::vision::FaceDetectionResult> result;
det.predict(img, &result);
// result[i]: {box, landmarks[5]}

// 人脸识别流水线（检测+特征）
auto rec = modeldeploy::vision::face::FaceRecognizerPipeline("det.onnx", "rec.onnx", option);
rec.set_det_threshold(0.5);
rec.cls_batch_size = 8;
```

**示例**：`examples/demo_face/`（demo_face_det / rec / age / gender / as_pipeline / rec_pipeline）

### 6.1 InsightFace（Buffalo 系列全流程）

`InsightFaceAnalysis` 从模型目录加载 InsightFace Buffalo 全家桶（det + 2d106 + 3d68 + recognition），一次 `analyze` 输出检测框/关键点/106/68 点/姿态/特征，适合人脸注册与比对。

```cpp
#include "modeldeploy/vision.h"

// 从 buffalo_l 模型目录加载全流水线
auto analysis = modeldeploy::vision::face::InsightFaceAnalysis::create_from_dir(
    "test_data/test_models/onnx/insightface/buffalo_l");
if (!analysis || !analysis->is_initialized()) return -1;

auto img = modeldeploy::ImageData::imread("test.jpg");
std::vector<modeldeploy::vision::face::InsightFaceResult> results;
if (!analysis->analyze(img, &results)) return -1;
// results[i]: {bbox, det_score, kps, landmark_2d_106, landmark_3d_68, pose, embedding}
```

也可单独使用子模型 `InsightFaceDet` / `InsightFaceGenderAge` / `InsightFaceLandmark` / `InsightFaceRecognition`（均位于 `modeldeploy::vision::face`，继承 `BaseModel`）。

**示例**：`examples/demo_face/demo_insightface_cxx.cpp`（`demo_insightface_cxx <model_dir> <image>`）

## 7. 车牌识别（License Plate）

| 模型类 | 功能 |
|--------|------|
| `LprDetection` | 车牌检测（框 + 关键点） |
| `LprRecognizer` | 车牌字符识别（字符串 + 颜色） |
| `LprPipeline` | 检测 + 识别串联 |

```cpp
auto lpr = modeldeploy::vision::lpr::LprPipeline("det.onnx", "rec.onnx", option);
std::vector<modeldeploy::vision::LprResult> result;
lpr.predict(img, &result);
// result[i]: {box, plate(字符串), color}
```

**示例**：`examples/demo_lpr/`

## 8. OCR（文字识别）

### 8.1 PaddleOCR（完整流水线）

```cpp
auto ocr = modeldeploy::vision::ocr::PaddleOCR(
    "det.onnx", "cls.onnx", "rec.onnx", option);
ocr.get_preprocessor().set_max_side_len(960);

std::vector<modeldeploy::vision::OCRResult> result;
ocr.predict(img, &result);
// result[i]: {text(识别文本), score, box(文本框四点), cls_label, cls_score}
```

### 8.2 单模块

| 模型类 | 功能 |
|--------|------|
| `DBDetector` | 文本检测（DB，输出多边形） |
| `Recognizer` | 文本识别（CTC） |
| `Classifier` | 方向分类（0°/180°） |

### 8.3 表格结构

| 模型类 | 功能 |
|--------|------|
| `StructureV2Layout` | 版面分析 |
| `StructureV2SERViLayoutXLMModel` | 语义实体识别 |
| `StructureV2Table` | 表格结构识别（SLANet） |
| `PPStructureV2Table` | 表格流水线 |

```cpp
auto table = modeldeploy::vision::ocr::PPStructureV2Table(
    "det.onnx", "rec.onnx", "table.onnx", option);
std::vector<modeldeploy::vision::OCRResult> result;
table.predict(img, &result);
```

**示例**：`examples/demo_ocr/`

### 8.4 公式识别（Formula → LaTeX）

`FormulaRecognizer` 输入裁剪后的公式图，输出 LaTeX 字符串，需字符/词表 dict（CTC `[batch, seq, num_class]` 输出）。常用于文档理解管线一键出 LaTeX。

```cpp
auto formula = modeldeploy::vision::ocr::FormulaRecognizer(
    "formula.onnx", "formula_dict.txt", option);
std::string latex;
if (formula.predict(im, &latex)) {
    std::cout << latex << std::endl;  // 如 "\frac{a}{b}"
}
```

**示例**：文档理解整体流程见 `examples/demo_doc/demo_doc.cpp`。

## 9. 行人属性（Pedestrian Attribute）

### 9.1 PedestrianAttribute

检测行人 + 多标签属性分类串联。

```cpp
auto attr = modeldeploy::vision::PedestrianAttribute("det.onnx", "cls.onnx", option);
attr.set_det_threshold(0.5);
attr.set_det_input_size({1280, 1280});
attr.set_cls_input_size({192, 256});
attr.cls_batch_size = 8;

std::vector<modeldeploy::vision::AttributeResult> result;
attr.predict(img, &result);
```

**示例**：`examples/demo_pipeline/demo_pedestrian_attribute_cxx.cpp`

## 10. 语音识别（ASR）

### 10.1 SenseVoice（流式识别）

```cpp
auto asr = modeldeploy::audio::asr::SenseVoice("sense_voice.onnx", option);
std::string text = asr.predict(wav_data);
```

### 10.2 AAsr（VAD + 识别流水线）

```cpp
auto asr = modeldeploy::audio::AAsr("vad.onnx", "sense_voice.onnx", option);
asr.predict(pcm_data, [](const std::string& text){ /* 回调 */ });
```

**示例**：`examples/demo_audio/demo_sense_voice_cxx.cpp`

### 10.3 ParaformerStreamingAsr（流式，仅 C++）

基于 Paraformer 的流式识别（在线 FBank + chunk 解码），分块喂波形、逐块解码输出文本。

```cpp
modeldeploy::audio::asr::ParaformerStreamingAsr asr(
    "encoder.onnx", "decoder.onnx", "tokens.txt", /*sample_rate=*/16000, /*num_threads=*/2);

asr.reset();
asr.accept_waveform(samples0);          // 可多次喂入 16k float 样本块
asr.accept_waveform(samples1);

modeldeploy::audio::asr::StreamingAsrResult r;
bool new_tok = asr.decode(/*is_final=*/false, &r);   // 逐步解码
// r.text 为本步新增文本；asr.text() 为累计全文

asr.input_finished();
asr.decode(/*is_final=*/true, &r);      // flush 末尾
```

> **当前仅 C++**（无 pybind / 独立示例 demo）。

**后端约束（A3 结论）**：流式 Paraformer **基于 ORT**。其 decoder 为**状态化**模型（encoder/decoder 双 Runtime，
decoder 每步携带上一块的 hidden state 往返），多后端未实现。针对 MNN 的评估结论：MNN 对像 decoder 这般
每块由应用自持状态 Tensor 并显式往返的状态化图块缺乏等价、可移植的表达（onnx→MNN 转换对包含动态
`com.microsoft` 风格状态输入/输出的图不保证保真），因此**不支持 MNN 后端**。使用非 ORT 后端（如
`RuntimeOption` 传入 `use_mnn_backend()`）会在**加载任何模型之前**明确报错并返回初始化失败（绝不静默钳制
为 ORT，也不会产生不可诊断的 ORT 加载混淆错误）：

```
ParaformerStreamingAsr(encoder) 仅支持 ORT 后端(状态化 decoder 多后端未实现);
请调用 runtime_option.use_ort_backend()。当前 backend=<Backend 枚举值>
```

如需自定义 encoder 运行时选项，可使用 `RuntimeOption` 构造重载：

```cpp
modeldeploy::RuntimeOption ro;
ro.use_ort_backend();                      // 必须是 ORT，否则初始化明确失败
ro.set_model_path("encoder.onnx");
ro.set_cpu_thread_num(2);
modeldeploy::audio::asr::ParaformerStreamingAsr asr(ro, "decoder.onnx", "tokens.txt");
```

## 11. 语音合成（TTS）

### 11.1 Kokoro

```cpp
auto tts = modeldeploy::audio::tts::Kokoro("kokoro.onnx", option);
auto audio = tts.predict("你好，世界");
audio.save_wav("out.wav");
```

支持中英混读（jieba 分词 + 文本正则化）。

**示例**：`examples/demo_audio/demo_kokoro_cxx.cpp`

**GPU 加速**：Kokoro 走通用 `RuntimeOption` → ORT 后端，`option.set_device(Device::GPU, 0)` 即可用 CUDA EP，代码无需改动；实测 GPU RTF≈0.32（对实时合成无压力，且不受 CPU 负载影响），比同负载 CPU 快 3–5 倍（CPU 空闲约 0.19）。

**模型变体实测结论**：
- int8 动态量化（MatMul-only，`onnxruntime.quantization`）对 kokoro-zh **几乎无提速**（~0.162→0.159）且改变输出长度/行为（量化漂移）→ 不建议。
- 本地 fp16 转换（`onnxconverter-common convert_float_to_float16`）会产生类型不一致（Albert Cast 输出 fp16 与后续 fp32 期望不匹配）→ 不可用。
- 如需 fp16/int8，请以官方导出版为准（如 `hexgrad/Kokoro-82M-v1.1-zh` 的 onnx 变体），并核对词表/token 与 `tokens.txt`、`voices.bin` 一致再替换 `model.onnx`。
- kokoro-zh 含 `SplitToSequence` 算子，**TensorRT 不支持**，无法转 TRT engine（有 CUDA 时用 ORT GPU EP 即可）。

### 11.2 统一流式合成（predict_stream）

Kokoro 实现 `ITtsModel::predict_stream(text, voice, speed, chunk_frames, cb)`：`chunk_frames == 0` 等价一次性合成（单次回调整段）；`> 0` 时按块回调，`cb` 返回 `false` 立即中止。

```cpp
tts.predict_stream("你好，世界。", "demo", 1.0f, 120,
    [](const float* samples, int n, float progress) -> bool {
        // samples: 当前块（n 个 mono float 采样），progress: [0,1]
        return true;
    });
```

**示例**：`examples/demo_audio/demo_tts_stream_cxx.cpp`（Kokoro 流式，`chunk_frames>`120 字数触发多块，回调打印 progress 与块长）。

## 12. VAD（语音活动检测）

### 12.1 SileroVAD

```cpp
auto vad = modeldeploy::audio::vad::SileroVAD("silero_vad.onnx", option);
std::vector<std::pair<int,int>> segments = vad.predict(pcm_data, sample_rate);
```

支持 16k/8k 采样率，32/64/96ms 窗口。

## 13. 模型与测试数据

- **测试模型**：`test_data/test_models/`（yolo11n 系列、人脸、OCR 等）
- **测试图片**：`test_data/test_images/`
- **测试数据下载**：见 AGENTS.md（需从 modelscope 单独下载）

## 14. 模型选择与转换

| 场景 | 推荐 |
|------|------|
| 通用检测 | YOLO11n/s/m（`.onnx`） |
| 分割 | YOLO11n-seg |
| 姿态 | YOLO11n-pose |
| 旋转框 | YOLO11n-obb |
| GPU 加速 | 转 `.engine`（TRT） |
| TPU 部署 | 转 `.bmodel`（Sophgo） |

各模型导出/转换教程见 [模型转换与量化](./conversion.md) 与 [后端详解](./backends.md)。

## 15. 多目标跟踪（Tracking）

在检测结果基础上做跨帧目标跟踪，输出稳定 `track_id`。基础类 `BaseTracker`，实现 `ByteTracker` / `BoTSORT` / `StrongSORT`。

```cpp
// 头文件: #include "modeldeploy/tracking.h"
// 用法见示例: examples/demo_tracking/demo_tracking_ort_cpu.cpp
```

- `ByteTracker`：轻量、适合实时；`BaseTracker` 提供 `update(detections)` 返回带 track_id 的轨迹。
- 与 `UltralyticsDet` 配合：det → track → 可视化，跨帧保持稳定 `track_id`。
- **示例**：`examples/demo_tracking/`（`demo_tracking_ort_cpu.cpp` 演示 det→track→可视化整体流程）。

## 16. 视频动作识别（Video Action Recognition）

基于视频帧序列的动作分类。两类模型：

- **TSN**（RGB 帧，配合 `VideoDecoder` 抽帧）：`vision::action::TSN`
- **ST-GCN**（骨架，`UltralyticsPose` 提关键点后输入）：`vision::action::StGcn`

```cpp
// TSN: examples/demo_action/demo_action.cpp
// ST-GCN 骨架: examples/demo_action/demo_action_skeleton.cpp
```

- **示例**：`examples/demo_action/`（输入 mp4 视频，输出 top 动作 label+score）。

## 17. 文档理解（Document Understanding → Markdown）

版面分析 `StructureV2Layout` 定位版面/公式/表格，配合 OCR 与表格识别输出整页 Markdown：
公式以 `$...$`、表格以 HTML 呈现。

```cpp
// 完整管线见: examples/demo_doc/demo_doc.cpp
// 用法: demo_doc <layout.onnx> <image> [<formula.onnx> [dict]] [--ocr ...] [--table ...]
```

- **示例**：`examples/demo_doc/demo_doc.cpp`。

## 18. 行人 Re-ID（Person Re-Identification）

`vision::reid::ReID` 基于 OSNet 输出 512-d 行人特征，配合内存 `ReIdGallery` 做检索匹配。

```cpp
// 用法: examples/demo_reid/demo_reid.cpp <model> <imgA> <imgB>
```

- **示例**：`examples/demo_reid/demo_reid.cpp`（输出 embedding 维度 + gallery 匹配 label/score）。

## 19. 声纹验证（Speaker Verification）

ECAPA-TDNN 输出 192-d 说话人 embedding，配合内存 `SpeakerGallery` 验证/检索。

```cpp
// 纯音频，无需 OpenCV。用法: examples/demo_speaker/demo_speaker.cpp <model.onnx> <wavA> <wavB>
```

- **示例**：`examples/demo_speaker/demo_speaker.cpp`（输出两段语音 embedding 维度 + gallery 匹配 label/score）。

## 20. 音频解决方案（音频方案）

提供常用音频场景的预组装方案：

| 方案 | 头文件 | 说明 |
|------|--------|------|
| 说话人分段 `SpeakerDiarization` | `audio/solutions/speaker_diarization.h` | VAD 切段 |
| 说话人检索 `SpeakerSearch` | `audio/solutions/speaker_search.h` | 声纹检索 |
| 流式识别 `StreamingStt` | `audio/solutions/streaming_stt.h` | 分块 push + 回调 |
| TTS 批处理 `TtsBatcher` | `audio/solutions/tts_batcher.h` | enqueue/dequeue_all |

> 各方案详细文档（应用场景 / 使用方式 / 算法原理）见 [解决方案总览](./solutions.md)。

**示例**：`examples/demo_audio_solutions/demo_diarization.cpp`、`demo_stream_stt.cpp`、`demo_tts_batch.cpp`。

## 21. NLP（jieba 分词 / 分类）

基于 jieba 的中文处理工具与可选 BERT 文本分类（ONNX）。

```cpp
// 纯工具: Splitter / Keywords / Stats / Tokenizer / Normalizer
// 文本分类: nlp::TextClassifier("bert.onnx", option)
// 用法: examples/demo_nlp/demo_nlp.cpp [bert.onnx] [text]
```

**示例**：`examples/demo_nlp/demo_nlp.cpp`。

## 22. 条码 / 二维码（Barcode / QR）

`vision::barcode::BarcodeDetector` 纯 CV 识别（零 DNN，跨全部后端），输出格式、文本、分数与是否二维码。

```cpp
// 用法: examples/demo_barcode/demo_barcode.cpp
// 输出形如: [QR Code] https://example.com/MD (score, is_qr)
```

**示例**：`examples/demo_barcode/demo_barcode.cpp`。

## 23. 手部关键点 / 关键点扩展

- **手部关键点** `vision::hand::HandKeypoint`：检测手 + 关键点。示例 `examples/demo_hand/demo_hand.cpp`。
- **关键点扩展**：车辆关键点、面部 Landmark 106 点。示例 `examples/demo_landmark/demo_landmark.cpp`（`demo_landmark <vehicle.onnx|none> <face.onnx|none> <image.jpg>`）。

## 24. CV 解决方案（场景方案）

基于 `SolutionBase` 的预组装视觉场景方案，无需自带权重、多为纯算法：

| 方案 | 类 | 说明 |
|------|----|------|
| 跨线计数 | `ObjectCounter` | 统计 line_in/line_out 与类别计数 |
| 热力图 | `Heatmap` | 生成密度热力峰 |
| 测速 | `SpeedEstimator` | 估算移动速度 m/s |
| 车位管理 | `ParkingManager` | 车位占用判定 |
| 距离估计 | `DistanceEstimator` | 基于水平面的像素→距离估计 |
| 目标模糊 | `ObjectBlur` | 对指定目标做马赛克/模糊 |
| 目标裁剪 | `ObjectCropper` | 按检测框裁剪目标图 |
| 针孔/鹰眼 | `VisionEye` | 基于视平线的透视变换可视化（`VisionEye(eye_level_y)`） |
| 健身动作计数 | `WorkoutMonitor` | 锻炼动作计数/监测 |

> 以上均为纯 C++ 解决方案层（`csrc/vision/solutions/`）；`DistanceEstimator / ObjectBlur / ObjectCropper / VisionEye` 暂无独立 demo。通用工具类（`Annotator / Detections / Metrics / InferenceSlicer / DetectionSmoother / Zone`）见 `csrc/vision/tools/`。

> 各方案与工具的详细文档（应用场景 / 使用方式 / 效果 / 原理 / 算法）见 [解决方案总览](./solutions.md) 与 [工具总览](./tools.md)。

**示例**：`examples/demo_solutions/demo_solutions.cpp`（ObjectCounter/Heatmap/SpeedEstimator/ParkingManager）；CV 纯工具（Annotator/LineZone/PolygonZone/Metrics mAP）见 `examples/demo_tools/demo_tools.cpp`。

## 25. 语义分割（Semantic Segmentation）

`UltralyticsSem` 输出逐像素类别，配合 Cityscapes 调色板可视化。

```cpp
auto m = modeldeploy::vision::detection::UltralyticsSem("yolo26n-sem.onnx", option);
auto im = modeldeploy::ImageData::imread("test_sem_540.jpg");

modeldeploy::vision::SemSegResult res;
m->predict(im, &res);   // res: {labels, shape, num_classes}
auto vis = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, true);
vis.imwrite("sem_out.jpg");
```

**示例**：`examples/demo_sem/`（`demo_sem_ort_cpu.cpp` 等全后端矩阵）

## 26. 深度估计（Depth Estimation）

`UltralyticsDepth` 输出逐像素单目深度（log 空间经 `exp` 还原为米），`vis_depth` 用 JET 伪彩色。

```cpp
auto m = modeldeploy::vision::detection::UltralyticsDepth("yolo26n-depth.onnx", option);
modeldeploy::vision::DepthResult res;
m->predict(im, &res);   // res: {depth, shape}
auto vis = modeldeploy::vision::vis_depth(im, res, true, false);
vis.imwrite("depth_out.jpg");
```

**示例**：`examples/demo_depth/`（全后端矩阵）

## 27. 文本正则化（ITN / Inverse Text Normalization）

把数字/量词等口语化内容转写为文字（ASR 后处理常用）。

```cpp
// modeldeploy::audio::tool
modeldeploy::audio::tool::InverseTextNormalizer itn;
std::string out = itn.normalize("2024年3月5日");   // 输出中文数字/量词文字
```

> **当前仅 C++**（无 pybind / 示例 demo）。

**WeTextProcessing 后端（可选，精度更高）**：`ENABLE_WETEXT=ON` 时可用
`modeldeploy::audio::tool::ItnBackend::WeText`（覆盖数字/日期/金额等口语化更全）。
依赖 wenet-e2e/WeTextProcessing + OpenFst + glog，构建时需
`-DWETEXT_INCLUDE_DIR=<WeText根> -DOPENFST_INCLUDE_DIR -DOPENFST_LIB -DGLOG_INCLUDE_DIR`；
运行时通过环境变量 `MODELDEPLOY_WETEXT_DIR` 给出含 `tagger.fst`/`verbalizer.fst` 的模型目录。
**标准 OpenFst（`kkm000/openfst` CMake 版）可用，勿用 csukuangfj fork（其头依赖 `dlfcn.h`，Windows 编不了）**：
给 `fst/string.h` 打 2 行最小补丁——`StringCompiler`/`StringPrinter` 构造加默认 `token_type = BYTE`——
即可在 Windows/MSVC 完整编译并链接进 SDK（本机已验证）。唯一跨平台的剩余依赖是 `tagger.fst`/`verbalizer.fst`
语法模型：需用 pynini/OpenFst 工具链在 Linux 编译生成后拷入；模型缺失时自动退化到内置轻量实现。

## 28. 视频解码（VideoDecoder）

FFmpeg 软解（h264/hevc → NV12），供视频动作识别等场景逐帧读取。

```cpp
// modeldeploy::video
modeldeploy::video::VideoDecoder dec;
if (dec.open("test.mp4")) {
    modeldeploy::vision::ImageData frame;   // NV12 CPU
    uint64_t pts_ms = 0;
    while (dec.next(&frame, &pts_ms)) {
        // 处理每一帧；dec.width()/height()/fps() 可得元信息
    }
    dec.close();
}
```

> 需要 `BUILD_VIDEO=ON`（FFmpeg）。示例见 `examples/demo_action/demo_action.cpp`（配合 TSN 动作识别）。

## 29. 轻量分割一切（FastSAM）

`FastSam` 复用 `InstanceSegResult` 消费路径，一次性输出 box + mask（对齐 `UltralyticsSeg`），
后处理产出 `InstanceSegResult{box, mask(二值图), label_id, score}`。

```cpp
auto m = modeldeploy::vision::seg::FastSam("fastsam-s.onnx", option);
m.get_preprocessor().set_size({1024, 1024});

std::vector<modeldeploy::vision::InstanceSegResult> result;
m.predict(img, &result);
// result[i]: {box, mask(二值图), label_id, score}
```

mask 以 `Mask` 结构保存（shape `{h, w}`，uint8 0/1），可用 `vis_iseg` 可视化。

`FastSam` 还提供 `predict_with_prompts`，在一次性全量输出（Everything）的基础上按提示词过滤实例，
**不重跑网络**：

- `bboxes`（`Rect2f` x/y/w/h，原图像素）：每个框取 IoU 最大的实例。
- `points` + `point_labels`（等长，`1`=前景保留 / `0`=背景剔除）：按掩码是否命中该点保留/剔除实例。
- 提示词为空时等价于全量 `predict`。

```cpp
modeldeploy::vision::seg::FastSamPrompts prompts;
prompts.bboxes.push_back(modeldeploy::vision::Rect2f(100.f, 80.f, 220.f, 180.f)); // 取最匹配该框的实例
prompts.points.push_back(modeldeploy::vision::Point2f(150.f, 130.f));
prompts.point_labels.push_back(1); // 前景：保留掩码命中该点的实例
std::vector<modeldeploy::vision::InstanceSegResult> result;
m.predict_with_prompts(img, prompts, &result);
```

Python 侧使用同名的值对象 `FastSamPrompts`（`bboxes` = `Rect2f[]`、`points` = `Point2f[]`、
`point_labels` = `int[]`），`predict_with_prompts(image, prompts)` 收单对象参数：

```python
import modeldeploy.vision as mv
from modeldeploy.vision import FastSamPrompts, Rect2f, Point2f

m = mv.FastSam("fastsam-s.onnx", option)
prompts = FastSamPrompts()
prompts.bboxes = [Rect2f(100, 80, 220, 180)]   # x,y,w,h；取最匹配该框的实例
prompts.points = [Point2f(150, 130)]
prompts.point_labels = [1]                     # 1=前景保留 / 0=背景剔除
result = m.predict_with_prompts(image, prompts)
```

> 各语言绑定同名可用：Python `FastSam.predict_with_prompts(image, prompts)`、C API `md_fastsam_predict_with_prompts`、
> Rust `FastSam.predict_with_prompts`、C# `FastSamModel.PredictWithPrompts`。
> MobileSAM 的两段式（编码器+解码器）本轮未接入。

**示例**：`examples/demo_sam/demo_fastsam.cpp`

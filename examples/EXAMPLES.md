# ModelDeploy Examples 目录

本文件列出 `examples/` 下**已在本机（CPU/ORT）验证可编译 + 可实际推理**的 demo。
构建（CPU，`BUILD_CAPI=ON`、`BUILD_AUDIO=ON`、`BUILD_VISION=ON`）：

```bash
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=ON \
      -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DENABLE_MNN=OFF -DENABLE_ORT=ON \
      -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_BENCHMARK=OFF
cmake --build build
```

运行：所有 demo 从 `build/bin` 启动（镜像内 `../../test_data` 相对该 cwd 解析到仓库根）。
测试数据需先从 modelscope 下载（见根 README / AGENTS.md）。

---

## C API（capi）

> `_capi`（capi）示例统一使用 `#include "../capi_common.h"` 的 `die()` 做错误处理。

| Demo | 类别 | 模型 | 输入 | 输出 |
|------|------|------|------|------|
| `demo_detection_capi` | 检测 | `onnx/yolo11n/yolo11n.onnx` | `test_images/test_detection0.jpg` | 打印检测框 + `capi_detection_out.jpg` |
| `demo_pose_capi` | 姿态 | `onnx/yolo11n/yolo11n-pose.onnx` | 人脸/行人图 | 关键点 + `capi_pose_out.jpg` |
| `demo_obb_capi` | 旋转框 | `onnx/yolo11n/yolo11n-obb_nms.onnx` | `test_images/test_obb1.jpg` | 旋转框 + `capi_obb_out.jpg` |
| `demo_instance_seg_capi` | 实例分割 | `onnx/yolo11n/yolo11n-seg_nms.onnx` | 图 | 分割实例 + `capi_iseg_out.jpg` |
| `demo_classification_capi` | 分类 | `onnx/yolo11n/yolo11n-cls.onnx` | `test_images/bus.jpg` | top5 + `capi_cls_out.jpg` |
| `demo_face_det_capi` | 人脸检测 | `onnx/face/scrfd_*.onnx` | `test_images/test_face1.jpg` | 人脸框/关键点 + `capi_face_det_out.jpg` |
| `demo_face_age_capi` | 年龄 | `onnx/face/age_predictor.onnx` | `test_images/test_face_id1.jpg` | 打印 age |
| `demo_face_gender_capi` | 性别 | `onnx/face/gender_predictor.onnx` | `test_images/test_face_gender.jpg` | 打印 gender |
| `demo_face_rec_capi` | 人脸特征 | `onnx/face/face_recognizer.onnx` | `test_images/test_face_id4.jpg` | 打印 1024 维 embedding |
| `demo_face_rec_pipeline_capi` | 检测+特征 | `scrfd_*.onnx|face_recognizer.onnx` | `test_images/test_face_detection4.jpg` | 打印每张脸 embedding |
| `demo_face_as_first_capi` | 防伪(first) | `onnx/face/fas_first.onnx` | `test_images/test_face_id3.jpg` | 打印 REAL/SPOOF |
| `demo_face_as_second_capi` | 防伪(second) | `onnx/face/fas_second.onnx` | `test_images/test_face_as_second2.jpg` | 打印 REAL/SPOOF |
| `demo_face_as_pipeline_capi` | 防伪管线 | `scrfd_*.onnx|fas_first.onnx|fas_second.onnx` | `test_images/test_face_detection4.jpg` | 逐脸打印 REAL/FUZZY/SPOOF |
| `demo_ocr_capi` | OCR 整链 | `ocr/ppocrv4_mobile/det_infer.infer|cls_infer.onnx|rec_infer.onnx|ppocrv4_dict.txt` | `test_images/test_ocr.png` | 打印行文本 + `capi_ocr_out.jpg`（演示 `det_db_box_thresh`/`cls_thresh` 参数） |
| `demo_lpr_pipeline_capi` | 车牌管线 | `lpr/yolov5plate.onnx|plate_recognition_color.onnx` | `test_images/test_lpr_pipeline.jpg` | 打印车牌/颜色 + `capi_lpr_out.jpg` |
| `demo_pedestrian_attribute_capi` | 行人属性 | `zhgd_det.onnx|zhgd_ml.onnx` | `test_images/test_pedestrian_attribute1.jpg` | 打印属性分数 + `capi_attr_out.jpg`（演示 `det_threshold`/`set_input_size`/`set_cls_input_size`） |
| `demo_kokoro_capi` | TTS | `kokoro_v1_1/model.onnx|tokens.txt|lexicon-gb-en.txt|lexicon-zh.txt|voices.bin|dict|test_data` | 合成文本 | `capi_tts_out.wav` |
| `demo_sense_voice_capi` | ASR | `sense_voice/model.int8.onnx|tokens.txt` | `sense_voice/test_wavs/zh.wav` | 打印识别文本 |
| `demo_image_from_base64` | 图像工具 | —（`test_base64_image.txt`） | — | `capi_base64_out.png` |
| `demo_image_from_bgr24` | 图像工具 | —（合成 BGR 样本） | — | `capi_bgr24_out.png` |
| `demo_image_rotate` | 图像工具 | `test_images/test_face_as_second.jpg` | — | `capi_rotate_original.jpg` / `capi_rotate90_out.jpg` |

> **防伪 kind**：`MD_MODEL_FACE_AS` = `SeetaFaceAsFirst`（被动防伪），`MD_MODEL_FACE_AS_SECOND` = `SeetaFaceAsSecond`，`MD_MODEL_FACE_AS_PIPELINE` = `scrfd|first|second` 管线。结果经 `md_result_spoof(res, i, &label)` 读取，label 0=REAL / 1=FUZZY / 2=SPOOF。

> **设备选择**：除 `md_option_set_device(opt, MD_DEV_* )` 外，可用 `md_option_set_device_id(opt, id)` 指定多卡/多 TPU 的设备号（GPU/TPU 生效，CPU 忽略；默认 0）。

---

## C++ SDK（`.cxx`）

已统一为 **CPU/ORT + 落盘保存**（`imshow` → `imwrite`，去掉 GPU/TRT 选项）：

| Demo | 类别 | 落盘输出 |
|------|------|----------|
| `demo_detection_cxx` | 检测（benchmark 循环） | `result_out.jpg` |
| `demo_detection_batch` | 检测（batch） | 打印各行目标数 |
| `demo_detection_multi_thread` | 检测（多线程） | 打印各线程目标数 |
| `demo_detection_multi_thread_trt` | 检测（多线程，CPU 版） | 打印各线程目标数 |
| `demo_multi_thread_compare` | 单/多线程对比 | 打印 FPS |
| `demo_benchmark` | 基准 | 打印 FPS |
| `demo_profile` | 剖析 | `print_benchmark` |
| `demo_pose_cxx` | 姿态 | `pose_out.jpg` |
| `demo_keypoint_cxx` | 关键点 | `keypoint_out.jpg` |
| `demo_obb_cxx` | 旋转框 | `obb_out.jpg` |
| `demo_instance_seg_cxx` | 实例分割 | `iseg_out.jpg` |
| `demo_classification_cxx` | 分类 | `cls_out.jpg` |
| `demo_sem_cxx` | 语义分割 | `sem_out.jpg` |
| `demo_depth_cxx` | 深度 | `depth_out.jpg` |
| `demo_lpr_detection_cxx` | 车牌检测 | `lpr_detection_out.jpg` |
| `demo_lpr_recognizer_cxx` | 车牌识别 | 打印车牌 |
| `demo_lpr_pipeline_cxx` | 车牌管线 | `lpr_pipeline_out.jpg` |
| `demo_ocr_cxx` | OCR 整链 | `ocr_out.jpg` |
| `demo_ocr_det_cxx` | OCR 检测子模型 | `ocr_db_det_out.jpg` |
| `demo_ocr_rec_cxx` | OCR 识别子模型 | `ocr_rec_out.jpg` |
| `demo_structure_table_cxx` | 表格 | `structure_table_out.jpg` |
| `demo_pp_structure_table_cxx` | PP 表格 | `pp_structure_table_out.jpg` |
| `demo_face_det_cxx` | 人脸检测 | `face_det_out.jpg` + `align_face_<i>.jpg` |
| `demo_face_gender_cxx` | 性别 | `face_gender_out.jpg` |
| `demo_face_rec_pipeline_cxx` | 检测+特征 | 打印 embedding |
| `demo_pedestrian_attribute_cxx` | 行人属性 | `pedestrian_attr_out.jpg` |
| `demo_tracking_ort_cpu` | 多目标跟踪（det→track→可视化，ByteTracker + 对比 BoT-SORT） | `result_tracking_ort_cpu.jpg`（画框 + 稳定 track_id） |
| `demo_barcode` | 条码/二维码（纯 CV，零 DNN，跨全部后端） | 打印每码 `[FORMAT] text (score, is_qr)`，示例 `[QR Code] https://example.com/MD` |
| `demo_hand` | 手部关键点 | 打印检测到的手数、每手 box/score/关键点数 |
| `demo_reid` | 行人 Re-ID（OSNet，512-d embedding + 内存 ReIdGallery） | 打印两图 embedding 维度 + gallery 大小 + `match(imgB,1) -> label A/B score`（用法 `demo_reid <model> <imgA> <imgB>`） |
| `demo_speaker` | 声纹/说话人验证（ECAPA-TDNN，192-d embedding + 内存 SpeakerGallery；纯音频，无需 OpenCV） | 打印两支语音 embedding 维度 + gallery 大小 + `match(B,1) -> label score`（用法 `demo_speaker <model.onnx> <wavA> <wavB>`，B 后可跟更多 wav） |
| `demo_doc` | 文档理解（layout + 公式/OCR/表格 → Markdown） | 打印整页图转换后的 Markdown（公式 `$...$`、表格嵌 HTML）（用法 `demo_doc <layout.onnx> <image> [<formula.onnx> [dict]] [--ocr det cls rec dict] [--table det rec table rec_label table_char]`） |
| `demo_action` | 视频动作识别（TSN，RGB 帧，VideoDecoder 抽帧） | `onnx/tsn/*.onnx` | 视频 mp4 | 打印 top3 动作 label+score（用法 `demo_action <tsn.onnx> <video.mp4>`） |
| `demo_action_skeleton` | 骨架动作识别（ST-GCN，VideoDecoder + UltralyticsPose 提关键点） | `onnx/stgcn/*.onnx|pose.onnx` | 视频 mp4 | 打印动作 label+score（用法 `demo_action_skeleton <stgcn.onnx> <pose.onnx> <video.mp4>`） |

> **跟踪 demo**：无真实视频时用单张测试图（`test_detection1.jpg` 等）模拟多帧序列——把同一批检测框按帧做轻微确定性抖动连续送入追踪器，展示同一物体在帧间保持**稳定 track_id**。逐帧打印 track 数量，并统计"稳定物体数（每个物体跨帧只使用单一 track_id）"。

> sophgo 目标（`demo_*_sophgo`）仅在 `ENABLE_SOPHGO=ON` 且平台支持时构建；本机 CPU 构建不参与。

---

## C#（dotnet）

```bash
cd csharp
dotnet build ModelDeployExample/ModelDeployExample.csproj -c Debug
cd ModelDeployExample/bin/Debug/net9.0
./ModelDeployExample.exe
```

`Program.cs` 的 `Main` 依次运行：检测（含 `SetParam("conf_threshold"/"nms_threshold")` 与 `ParamNames()/ParamType()` 自省演示）→ 图像工具 → 分类 → 姿态 → OCR → InsightFace → ASR(SenseVoice) → TTS(Kokoro)。各模型推理结果打印到控制台，检测/图像可视化落盘 `detection_annotated.jpg` / `annotated.jpg`，TTS 落盘 `output.wav`。

> TTS 中文合成需要原生端按 UTF-8 封送字符串：`md_audio_tts` 的 `text`/`voice` 以 UTF-8 字节指针传递（`ModelDeploy/AudioModels.cs` 中 `Utf8` 手动封送）。

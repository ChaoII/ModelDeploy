# ModelDeploy 多语言 API

ModelDeploy 核心逻辑全部在 C++ SDK，提供 **C++ / Python / C / C# / Rust** 五种绑定，其余语言是对 C++/C 的薄封装，行为一致。

| 语言 | 文档 | 定位 |
|------|------|------|
| C++ | [cpp](./cpp.md) | 首选，完整功能全部模型/后端 |
| Python | [python](./python.md) | pybind11，科学计算场景 |
| C | [capi](./capi.md) | 嵌入式 / FFI 桥接（`md_*` 前缀） |
| C# | [csharp](./csharp.md) | .NET（`ModelDeploy` 命名空间） |
| Rust | [rust](./rust.md) | FFI 封装 C API |

后端/设备/精度配置在所有语言中保持一致，见 [RuntimeOption 配置](../runtime_option.md)。

## 跨语言模型/类名对照表

五种绑定的模型族类名/构造方式对照。每类模型在各语言下的写法如下（C API 为**统一分发点**，经 `md_model_create` 按 `MDModelKind` 创建）：

| 模型族 | C++ | Python | C API | C# | Rust |
|--------|-----|--------|-------|----|------|
| 目标检测 | `vision::detection::UltralyticsDet` | `modeldeploy.vision.UltralyticsDet` | `MD_MODEL_DETECTION`（经 `md_model_create`） | `Models.DetectionModel` | `UltralyticsDet` |
| 实例分割 / FastSAM | `vision::detection::UltralyticsSeg` / `vision::seg::FastSam` | `modeldeploy.vision.UltralyticsSeg` / `.FastSam` | `MD_MODEL_INSTANCE_SEG` / `MD_MODEL_FASTSAM`（经 `md_model_create`） | `Models.InstanceSegModel` / `Models.FastSamModel` | `UltralyticsSeg` / `FastSam` |
| 语义分割 | `vision::detection::UltralyticsSem` | `modeldeploy.vision.UltralyticsSem` | `MD_MODEL_SEM_SEG`（经 `md_model_create`） | `Models.SemSegModel` | `UltralyticsSem` |
| 深度估计 | `vision::detection::UltralyticsDepth` | `modeldeploy.vision.UltralyticsDepth` | `MD_MODEL_DEPTH`（经 `md_model_create`） | `Models.DepthModel` | `UltralyticsDepth` |
| 姿态 / 关键点 | `vision::detection::UltralyticsPose` / `vision::hand::HandKeypoint` / `vision::landmark::VehicleKeypoint` / `vision::landmark::FaceLandmark` | `modeldeploy.vision.UltralyticsPose` / `.HandKeypoint` / `.landmark.VehicleKeypoint` / `.landmark.FaceLandmark` | `MD_MODEL_POSE` / `MD_MODEL_HAND` / `MD_MODEL_VEHICLE_KEYPOINT` / `MD_MODEL_FACE_LANDMARK`（经 `md_model_create`） | `Models.PoseModel` / `Models.HandModel` / `Models.VehicleKeypointModel` / `Models.FaceLandmarkModel` | `UltralyticsPose` / `HandKeypoint` / `VehicleKeypoint` / `FaceLandmark` |
| OBB（旋转框检测） | `vision::detection::UltralyticsObb` | `modeldeploy.vision.UltralyticsObb` | `MD_MODEL_OBB`（经 `md_model_create`） | `Models.ObbModel` | `UltralyticsObb` |
| 图像分类 | `vision::classification::Classification` | `modeldeploy.vision.Classification` | `MD_MODEL_CLASSIFICATION`（经 `md_model_create`） | `Models.ClassificationModel` | `Classification` |
| OCR（PaddleOCR + 子模型） | `vision::ocr::PaddleOCR`（子模型 `DBDetector` / `Recognizer` / `Classifier`） | `modeldeploy.vision.PaddleOCR`（子模型 `DBDetector` / `Recognizer` / `Classifier`） | `MD_MODEL_OCR`（子模型 kind `MD_MODEL_OCR_DET` / `OCR_REC` / `OCR_CLS`） | `Models.OcrModel`（子模型 `DbDetectorModel` / `RecognizerModel` / `OcrClassifierModel`） | `PaddleOCR`（子模型 `DbDetectorModel` / `RecognizerModel` / `OcrClassifierModel`） |
| OCR 进阶（版面/表格/公式/文档） | `vision::ocr::StructureV2Layout` / `StructureV2Table` / `PPStructureV2Table` / `FormulaRecognizer` / `DocToMarkdown` | `modeldeploy.vision.StructureV2Layout` / `StructureV2Table` / `PPStructureV2Table` / `FormulaRecognizer` / `DocToMarkdown` | 仅 `MD_MODEL_FORMULA_RECOGNIZER`（公式，`md_result_formula`）；版面/表格/文档未绑定 | 仅 `Models.FormulaRecognizerModel`（公式）；版面/表格/文档未绑定 | 仅 `FormulaRecognizer`（公式）；版面/表格/文档未绑定 |
| 人脸（Scrfd / SeetaFace 族） | `vision::face::Scrfd` / `SeetaFaceID` / `SeetaFaceAge` / `SeetaFaceGender` / `SeetaFaceAsFirst` / `SeetaFaceAsSecond` / `SeetaFaceAsPipeline` / `FaceRecognizerPipeline` | `modeldeploy.vision.Scrfd` / `SeetaFaceID` / `SeetaFaceAge` / `SeetaFaceGender` / `SeetaFaceAsFirst` / `SeetaFaceAsSecond` / `SeetaFaceAsPipeline` / `FaceRecognizerPipeline`（全绑定） | `MD_MODEL_FACE_DET` / `FACE_REC` / `FACE_AGE` / `FACE_GENDER` / `FACE_AS_PIPELINE` / `FACE_REC_PIPELINE`（经 `md_model_create`） | `Models.FaceDetModel` / `FaceRecModel` / `FaceAgeModel` / `FaceGenderModel` / `FaceRecognizerPipelineModel`；防伪（`SeetaFaceAs*`）未绑定 | `Scrfd` / `SeetaFaceID` / `SeetaFaceAge` / `SeetaFaceGender` / `FaceRecognizerPipelineModel`；防伪（`SeetaFaceAs*`）未绑定 |
| InsightFace 全流程 | `vision::face::InsightFaceAnalysis`（子模型 `InsightFaceDet` / `InsightFaceRecognition` / `InsightFaceLandmark` / `InsightFaceGenderAge`） | `modeldeploy.vision.InsightFaceAnalysis`（子模型 `InsightFaceDet` / `InsightFaceRecognition` / `InsightFaceLandmark` / `InsightFaceGenderAge`，全绑定） | `MD_MODEL_INSIGHTFACE` / `MD_MODEL_INSIGHTFACE_DET`（经 `md_model_create`） | `Models.InsightFaceModel` / `Models.InsightFaceDetModel` | `InsightFaceAnalysis` / `InsightFaceDetModel` |
| 车牌 LPR | `vision::lpr::LprPipeline`（子模型 `LprDetection` / `LprRecognizer`） | `modeldeploy.vision.LprPipeline`（子模型 `LprDetection` / `LprRecognizer`） | `MD_MODEL_LPR_PIPELINE`（子模型 kind `MD_MODEL_LPR_DET` / `LPR_REC`，经 `md_model_create`） | `Models.LprModel`（子模型 `LprDetectionModel` / `LprRecognizerModel`） | `LprPipeline`（子模型 `LprDetectionModel` / `LprRecognizerModel`） |
| 行人属性 + ReID | `vision::PedestrianAttribute`；`vision::reid::ReID` + `ReIdGallery` | `modeldeploy.vision.PedestrianAttribute` / `.ReID` / `.ReIdGallery`（全绑定） | `MD_MODEL_PED_ATTR` / `MD_MODEL_REID`（经 `md_model_create`）；`ReIdGallery` 未绑定 | `Models.PedestrianAttributeModel` / `ReIdModel`；`ReIdGallery` 未绑定 | `PedestrianAttribute` / `ReID`；`ReIdGallery` 未绑定 |
| 条码 + 多目标跟踪 | `vision::barcode::BarcodeDetector`；`vision::tracking::ByteTracker` / `BotSortTracker` / `StrongSortTracker` | `modeldeploy.vision.BarcodeDetector` / `ByteTracker` / `BotSortTracker` / `StrongSortTracker` | 独立句柄 `md_barcode_*`；`md_tracker_*`（`MDTrackerKind`，不属 `md_model_create`） | `BarcodeDetector`；`ModelDeploy.Tracking.Tracker`（`TrackerKind`） | `BarcodeDetector`；`Tracker`（`TrackerKind`） |
| CV 解决方案 + 工具 | `vision::solution::{ObjectCounter, Heatmap, RegionCounter, QueueManager, TrackZone, SpeedEstimator, ParkingManager, FallDetector, WorkoutMonitor, DistanceEstimator}` + `vision::tool::{Detections, LineZone, PolygonZone, iou, nms, from_track}` | `vision.solutions.*` / `vision.tools.*`（全绑定） | `md_solution_create(MD_SOLUTION_*)`；ObjectCounter/Heatmap/RegionCounter/Queue/Zone 有逐帧函数，Speed/Distance/Workout/Parking/Fall 仅创建 | `ModelDeploy.Solutions.{ObjectCounter, Heatmap, RegionCounter, QueueManager, TrackZone}` + `Tool.Iou`；其余未绑定 | `solution::{ObjectCounter, Heatmap, RegionCounter, QueueManager, TrackZone}` + `iou`；其余未绑定 |
| 音频（ASR / TTS / 声纹） | `audio::asr::SenseVoice`（ASR） / `audio::tts::Kokoro`（TTS，24k） / `audio::speaker_verify::SpeakerVerify` + `audio::SpeakerGallery` | `md.audio.SenseVoice` / `.Kokoro` / `.SpeakerVerify` / `.SpeakerGallery`（全绑定） | `MD_MODEL_ASR` / `MD_MODEL_TTS` / `MD_MODEL_SPEAKER_VERIFY`（经 `md_model_create`）+ `md_audio_asr*` / `md_audio_tts*` / `md_audio_speaker_embed`；声纹库用 `md_audio_speaker_search_*` | `Models.SenseVoiceModel` / `KokoroModel` / `SpeakerVerifyModel`；`SpeakerGallery` 未绑定，用 `Audio.SpeakerSearch` | `SenseVoice` / `Kokoro` / `SpeakerVerify` / `SpeakerGallery`（纯 Rust） |
| 音频 solutions + tools | `audio::solution::{SpeakerSearch, StreamingStt, TtsBatcher}` + `audio::tool::{InverseTextNormalizer, ItnEngine, Resampler, Fbank, Spectrum, Waveform}` | `audio.solutions.{SpeakerSearch, TTSBatcher}` / `audio.tools.{InverseTextNormalizer, ItnEngine, Fbank, Resampler, Spectrum, HotwordContext}`（全绑定） | `md_audio_solution_create(MD_AUDIO_SPEAKER_SEARCH)` + `md_audio_speaker_search_*` / `md_audio_resample`；TTSBatcher 仅可创建，ITN 未绑定 | `Audio.SpeakerSearch` + `Audio.Tools.Resample`；TTSBatcher / ITN 未绑定 | `audio::SpeakerSearch` + 自由函数 `resample`；ITN / TTSBatcher 未绑定 |

其余模型族在后续小节逐个补全。

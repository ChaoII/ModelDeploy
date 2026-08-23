# 视觉应用解决方案层（CV Solutions）—— 设计规范

- 日期：2026-08-23
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 11（新增，紧随 Item 4/5/7/6/8 之后）
- 核心目标：借鉴 Ultralytics Solutions / supervision，在既有 C++ SDK（已有检测/分割/姿态/跟踪 ByteTrack/视频解码/关键点/动作能力）之上，编排出一层**面向真实应用场景的解决方案** + 一层**可复用 CV 工具**，让 SDK 不再是空洞的底层推理壳，并充分发挥 C++ 性能。
- 领域扩展：本文件以 CV Solutions 为主（§1–§7 为 CV 的「方案层 + 工具层」规范），并在「扩展一/扩展二」中把**同一套分层理念**复制到 **Audio**（`audio::solution` / `audio::tool`）与 **NLP**（`nlp::tool` / `nlp::solution`），形成 **CV / Audio / NLP 三域结构**。三域共享同一设计原则：**方案层** = 组合既有模型做真实场景，**工具层** = 纯 C++ / 无模型依赖、可单测。

---

## 1. 背景与动机

现状（已具备并可复用的地基）：
- 检测/分割/姿态/分类：`vision::detection/segmentation/pose/...`（yolo26n 全套 + 真实权重已在本机）。
- 多目标跟踪：`vision::tracking::ByteTracker`（`base_tracker.h`：`Detection{box,score,label_id,feature}` → `update(detections, frame, timestamp) → vector<TrackResult{track_id,box,...}>`；已有 pybind `tracking_pybind.cpp`）。
- 视频解码：`video::VideoDecoder`（FFmpeg 软解 → NV12 ImageData，`BUILD_VIDEO`）。
- 关键点：`vision::landmark`（Face/Vehicle）、`hand`、pose 关键点。

因此本 Item 落点：**在既有能力之上，以 C++ 编排出应用解决方案层**。这些方案主要是「组合 + 少量算法逻辑」，几乎不需要新权重（复用 yolo/姿态/跟踪权重点），纯 C++、性能强。

该分层理念不限于视觉：Audio（已有 ASR SenseVoice / TTS Kokoro / VAD Silero / 声纹 SpeakerVerify / SpeakerGallery / text_normalize）与 NLP（jieba、文本归一化等轻量文本工具）同样适用「方案层 + 工具层」，下文「扩展一/扩展二」按同一范式补齐，使 SDK 三域统一、能力不空洞。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope（第一里程碑，按用户全选）
1. **ByteTracker 加固/可用性验证**（已有，重点是把 `Detection.feature`（ReID）接入、`update(timestamp)` 语义、状态机校验，作为地基）。可选补充 `SORT`/`IouTracker` 轻量跟踪器。
2. **Object Counting 行目标计数**：跨线 in/out 双向计数 + 区域(Region)计数 + 类维度计数（对齐 ObjectCounter/RegionCounter）。
3. **Heatmap 热力图**：基于跟踪轨迹的 ROI 停留/经过热度累加。
4. **Speed Estimation 测速**：track_id 跨帧质心位移 + 时间戳 → 速度（像素/秒 + 米/秒标定）。
5. **Distance Distance 距离**：质心欧氏距离 + 像素→米标定（两点/多对）。
6. **Object Cropping + Blurring 裁剪/模糊**：按框裁剪 ROI + 隐私高斯模糊原图区域。
7. **Workouts Monitoring 健身监测**：基于姿态关键点角度（肩-肘-腕）做动作计数（仰卧起坐/开合跳/深蹲雏形）。
8. **Parking Management 停车管理**：位元区域定义 + 垂直框占用判定 → available/filled slot。
9. **VisionEye 可视化映射**：质心到"眼点"连线轨迹可视化（用户已确认全做）。

### Out-of-Scope（明确不做，YAGNI）
- 不做 Streamlit 前端 / Analytics 图表渲染（偏前端，C++ 侧无价值）。
- 不做 Security Email 报警（依赖邮件服务，与应用无关）。
- 不做 Similarity Search-CLIP（需新 CLIP 权重 + 向量索引，超范围）。
- 不做 3D 测距 / 相机标定外参（除像素→米线性标定外不做相机模型）。
- 不做 supervision 的 Python 生态专属（converters 多源导入、PyTorch VLM、notebook、数据集 YOLO/COCO/VOC 互转）——超出 C++ SDK 价值，明确排除。

> 借鉴来源说明：Ultralytics Solutions 提供「业务方案」编排；supervision 提供「细粒度工具层」（标注器/几何/数据结构/指标）。本 Item 两者**都取**，形成「解决方案层（§3）+ 工具层（§4）」双结构，让 SDK 既有可落地场景，又有可复用工具，不空洞。

## 3. 架构与组件

新增 `csrc/vision/solutions/` 目录，命名空间 `modeldeploy::vision::solution`。每个解决方案是一组可组合的 C++ 类，输入 `vector<Detection/track/track-result>` + `ImageData`，输出业务结果与该方案的标注图层。

```
csrc/vision/solutions/
    solution_base.h            # 公共接口/标注画布/配置基类
    ObjectCounter.h/.cpp       # 跨线 + 区域 + 类维度计数
    Heatmap.h/.cpp             # 轨迹热度
    SpeedEstimator.h/.cpp      # 测速
    DistanceEstimator.h/.cpp   # 距离
    ObjectCropper.h/.cpp       # 裁剪
    ObjectBlur.h/.cpp          # 模糊
    WorkoutMonitor.h/.cpp      # 健身监测（姿态角度计数）
    ParkingManager.h/.cpp      # 停车管理
```

公共约定：
- 每个方案构造可配置（如 region 点集、标定比例、阈值、类别过滤）。
- `update(frame: const ImageData&, tracks: const vector<TrackResult>&, detections?, timestamp)` → 更新内部统计 + 返回方案结果（计数、速度、热度图、占用等）。
- `draw(frame*, ...)` 在帧上叠加可视化（复用现有可视化工具）。
- 纯内存/纯逻辑（除绘制的 opencv 外无模型依赖），可独立单测（无需权重）。

### 3.1 依赖复用
- `tracking::ByteTracker.update(detections, &frame, timestamp)` 提供 track_id。
- 计数/测速/热力图直接吃 `TrackResult`（track_id+box 稳定）。
- Workout 吃 pose 关键点（复用 `UltralyticsPose` 输出的 `KeyPointsResult`）。
- 裁剪/模糊、停车吃 `Detection` 框（不需跟踪）。
- 绘制复用现有 `vis_*` / opencv。

## 4. 工具层（supervision 借鉴，`csrc/vision/tools/`，命名空间 `modeldeploy::vision::tool`）

supervision 的细粒度工具层，全部纯 C++/OpenCV、无模型依赖、可独立单测。

```
csrc/vision/tools/
    detections.h/.cpp        # 统一 Detections 容器（box/class/conf/mask/track_id）+ IoU/NMS/Boxes/Polygons 工具
    annotator.h/.cpp         # Annotator 标注器族：BoundingBox/KeyPoint/Trace/Label/Blur/Pixelate/Ellipse/Halo/RoundBox + 组合
    zone.h/.cpp              # LineZone（跨线）/ PolygonZone（多边形区域）几何抽象
    metrics.h/.cpp           # mAP/Precision/Recall/F1 评测指标
    slicer.h/.cpp            # Inference Slicer 大图切片（小目标检测）
    smoother.h/.cpp          # Detection Smoother 检测抖动平滑
```

- **Detections**：统一容器 `Detections{ boxes[], class_id[], confidence[], masks?, tracker_id[] }`，配 `iou()`/`nms()`/`filter_by_class()`/`filter_by_zone()` 等工具——作为检测结果的标准载体，可被 Solutions 层共用。
- **Annotator**：`Annotator`（叠加画布，仿 supervision）+ `BoundingBoxAnnotator`/`KeyPointAnnotator`/`TraceAnnotator`(轨迹线)/`LabelAnnotator`/`BlurAnnotator`/`PixelateAnnotator`/`EllipseAnnotator`/`HaloAnnotator`/`RoundBoxAnnotator` + 组合器，绘制到 `ImageData`。是可视化最大增量。
- **Zone**：`LineZone{trigger_count, crossing_logic, reset()}`（跨线 in/out 计数基元）与 `PolygonZone{count_in_zone, contains(), current_count()}`（多边形区域计数/过滤基元），计数/停车/过滤复用。
- **Metrics**：`mAP/Precision/Recall/F1`（对齐 `common_values` + 混淆矩阵），供真实权重 benchmark 与测试判定。
- **Slicer**：`InferenceSlicer{slice_image()->(tiles, offsets), override_detections}`——大图切块 + 拼回，小目标检测。
- **Smoother**：`DetectionSmoother{update(detections)->smoothed}` 时序指数/EMA 抖动抑制。

## 4. Python（pybind）
- `csrc/pybind/vision/solutions_pybind.cpp`：`vision.solutions` 子模块，绑定各方案类（构造/配置/update/draw/结果读取）。
- `csrc/pybind/vision/tools_pybind.cpp`：`vision.tools` 子模块，绑定 Detections/IoU/NMS/Annotator/Zone/Metrics/Slicer/Smoother。
- 注册进 `vision_pybind.cpp`（append 到既有之后），BUILD_VISION 门控。

## 5. CAPI
- 新建轻量 `solution` C 风格句柄（`md_solution_object_counter` 等）：输入 track/detection 数组 → 输出结果。便于 C#/Rust 消费。
- **第一里程碑全 6 面一次做齐**（C++/Python/CAPI/C#/Rust/demo+tests，用户已确认）。

## 6. C# / Rust
- 薄封装（随 CAPI）。

## 7. demo + docs
- `examples/demo_solutions/`：一个综合 demo，读视频/摄像头 → 检测 + ByteTrack → 计数/测速/热力图叠加 → 窗口显示；Workout/Parking 各做独立小 demo。
- `examples/demo_tools/`（可选）：展示 Annotator/Zone/Metrics 工具用法。
- README/EXAMPLES.md 能力行加「CV Solutions 应用方案（计数/热力图/测速/停车/健身等）+ CV Tools（标注器/区域/评测）」。

---

# 扩展一：Audio 领域（方案层 + 工具层，`audio::solution` / `audio::tool`）

## A.1 背景与地基（CV 同范式）

Audio 侧已具备完整模型地基（均为 `BaseModel` 子类，走 ORT/MNN）：
- ASR：`audio::asr::SenseVoice`（`csrc/audio/asr/`）。
- TTS：`audio::tts::Kokoro`（`csrc/audio/tts/`）。
- VAD：`audio::vad::SileroVAD`（`csrc/audio/vad/`，`predict(data)->string` 逐帧状态机输出 speech/silence 段）。
- 声纹：`audio::speaker_verify::SpeakerVerify`（ECAPA-TDNN embedding）+ `audio::SpeakerGallery`（内存说话人库，`csrc/audio/`）。
- 文本：`audio::text_normalize`（数字/日期/电话/量词/英文归一化，`csrc/audio/text_normalize/`）。
- 已有流式组装参考：`audio::AAsr`（`asr_pipeline.h/.cpp`，VAD 门控 + SenseVoice 的 push/run 模式）。

Audio 落点与 CV 相同：在既有能力上编排出**方案层**（说话人日志 / 实时转写 / 声纹识别检索 / TTS 批处理）+ **工具层**（wav 读写、重采样、特征、可视化、VAD 分段、元信息）。Audio 方案仍以「组合 + 算法逻辑」为主，除声纹/SenseVoice/VAD 权重外**不引入新第三方依赖**（复用已捆绑 samplerate、kaldi-native-fbank、cppjieba）。

## A.2 Audio 工具层（`csrc/audio/tools/`，命名空间 `modeldeploy::audio::tool`）

```
csrc/audio/tools/
    wav_io.h/.cpp           # WavIO：16k/任意采样率 PCM 与 WAV 读写（RIFF/PCM/Float32，meta + samples）
    resampler.h/.cpp        # Resampler：重采样（复用已捆绑 samplerate 库）
    mfcc.h/.cpp             # Fbank+Mel 特征（复用 kaldi-native-fbank，供 ECAPA/声纹前端）
    waveform.h/.cpp         # Waveform + Spectrum 可视化（波形图 → 数组/图像，合成均可断言）
    vad_segment.h/.cpp      # VadSegment：基于 Silero 的连续音频分段（VAD 状态机 → [speech_segments]）
    audio_meta.h/.cpp       # AudioMeta：时长/频道/采样率/格式/位深读取
```

- **WavIO**：读 `WavData{meta, vector<float> samples}`、写 WAV；容错（缺头、非 PCM）。工具层输入输出底盘。
- **Resampler**：多采样率互转（8k/16k/44.1k/48k），复用 samplerate 的 sinc/linear；供 SenseVoice(16k)/ECAPA(16k)/任意输入对齐。
- **Mfcc/Fbank**：分帧(25ms/10ms)+加窗→ Mel-fbank（80 维可配）→ 可选 DCT→ MFCC；复用 kaldi-native-fbank，与 ECAPA preprocess 对齐。
- **Waveform/Spectrum**：把 samples → 波形坐标数组 / FFT 幅值谱（可视化可喂 `draw` 或导出为向量，单测可用合成正弦断言峰值频率）。
- **VadSegment**：`feed(samples)` / `segments() -> vector<Seg{start_ms,end_ms,samples}>`；封装 Silero 状态机，产出可消费语音段（供日志/STT 复用）。
- **AudioMeta**：解析 WAV 头返回 `meta{channels, sample_rate, bits, duration_ms, format}`，纯解析、无解码。

> 与 CV 相同：全部纯 C++/现有库、**无模型依赖**（VadSegment 例外调用 Silero，为可选组合），可独立单测。

## A.3 Audio 方案层（`csrc/audio/solutions/`，命名空间 `modeldeploy::audio::solution`）

```
csrc/audio/solutions/
    solution_base.h/.cpp       # 公共接口/回调/结果结构基类
    speaker_diarization.h/.cpp # SpeakerDiarization：VAD 分段 + SpeakerVerify embedding 聚类 → “谁在何时说话”
    streaming_stt.h/.cpp       # StreamingSTT：VAD 门控 + SenseVoice 流式转写（实时）
    speaker_search.h/.cpp      # SpeakerSearch：SpeakerVerify + SpeakerGallery 识别/检索闭环
    tts_batcher.h/.cpp         # TTSBatcher：文本队列 → 音频（Kokoro）批处理
```

- **SpeakerDiarization（说话人日志）**：`run(audio)` → VAD 切段 → 每段 SpeakerVerify embedding → 聚类（阈值相似度聚合/分层）→ `vector<Segment{start_ms,end_ms,speaker_id}>`。达成「谁在何时说话」。
- **StreamingSTT（实时转写 STT）**：`push(data, sr)` + `on_text(string)` 回调；VAD 门控累积语音段 → SenseVoice 转写 → 流式文本输出（参考 `AAsr` 的 push/run 模式）。
- **SpeakerSearch（声纹识别/检索）**：输入音频 → embedding → `SpeakerGallery.match` → `(label, score) top-k`；含 enroll 入口，形成「注册→识别→检索」闭环。
- **TTSBatcher（TTS 批处理）**：`enqueue(texts)` → 工作线程池调 Kokoro → `dequeue wavs`；文本→音频队列化批处理。

## A.4 Audio 六面交付
- **Python**：`csrc/pybind/audio/solutions_pybind.cpp`、`tools_pybind.cpp` → `audio.solutions` / `audio.tools` 子模块（注册进 main.cpp `#ifdef BUILD_AUDIO` 块）。
- **CAPI**：轻量 `md_audio_*` 方案句柄（如 `md_audio_diarization_*` / `md_audio_stt_*`）；工具函数（`md_audio_wav_read` / `md_audio_resample` / `md_audio_meta`）纯函数、易绑定。
- **C# / Rust**：薄封装随 CAPI；工具层函数式接口天然适合 P/Invoke / extern。
- **demo + docs**：`examples/demo_diarization/`、`demo_stream_stt/`、`demo_tts_batch/`；README/EXAMPLES 能力行加 Audio Solutions/Tools。**6 面一次做齐**（与 CV 一致，用户已确认）。

---

# 扩展二：NLP 领域（工具层 + 方案层，`nlp::tool` / `nlp::solution`）

## B.1 说明与前提

项目**无 BERT/LLM 大模型地基**（已确认），因此 NLP 不模仿 CV「复用已有模型组合方案」的思路，而是：**底层 = 轻量纯 C++ 文本工具**（jieba 分词、句子切分、文本归一化、关键词提取、文本统计）+ **上层 = 引入一个小型 ONNX 文本模型**做**一个**具体方案（YAGNI，不多引入）。它仍然是「方案层 + 工具层」结构：工具层无模型、可单测；方案层依赖一个小权重外链模型。

## B.2 NLP 工具层（新目录 `csrc/nlp/tools/`，命名空间 `modeldeploy::nlp::tool`）

```
csrc/nlp/tools/
    tokenizer.h/.cpp      # jieba 中文分词（复用已捆绑 cppjieba）/ 词形切分
    splitter.h/.cpp       # 句子切分（中英文标点断句）
    normalizer.h/.cpp     # 通用文本归一化（复用/抽取 audio/text_normalize 的数字/日期/单位逻辑或独立精简版）
    keywords.h/.cpp       # 关键词提取（启发式 TF / 停用词过滤）
    stats.h/.cpp          # 文本统计（字数/词数/句数/字频/分段）
```

- **Tokenizer**：包装 cppjieba（MP/HMM/Full 模式可选），输入 string → token 序列。
- **Splitter**：`句子切分`——按 。！？；换行等切句，返回带偏移的句子列表。
- **Normalizer**：抽取 `audio/text_normalize` 的数字/日期/电话/量词逻辑为通用接口（避免 NLP/音频重复实现），或提供精简适配层。
- **Keywords**：停用词 + 词频（TF）排序取 top-K，启发式、无模型。
- **Stats**：纯统计（中文字数/英文词数/句子数/字符频率/最长句等）。

> 全部纯 C++ 轻量、无大模型依赖，可独立单测。目录独立为 `csrc/nlp/`（而非塞进 audio），与三个域扁平对齐。

## B.3 NLP 方法层（`csrc/nlp/solutions/`，命名空间 `modeldeploy::nlp::solution`）—— 引入一个小型 ONNX 文本模型

### B.3.1 选型建议
候选与取舍：

| 任务 | 模型 | 体积 | 说明 | 决策 |
|------|------|------|------|------|
| **文本分类（情感/主题）** | DistilBERT / BERT 分类 ONNX | ~60–80MB | 通用、演示直观、ONNX 易得 | **推荐**（默认情感二类/三类，职责清晰） |
| 标点恢复（punctuation） | 小型 punctuation ONNX | ~30–60MB | 可复用于 ASR 后处理，但与文本工具耦合弱 | 备选 |
| 命名实体 / 复杂 NLU | 需预训练+词表 | 大 | 超范围，YAGNI | 排除 |
| 开放域对话 / 生成 | LLM 类 | 极大 | 无地基，明确排除 | 排除 |

**决策：引入「一个」小型 BERT 文本分类 ONNX（默认情感分类，主题可选），作为 NLP 域唯一带权重的方案**。理由：通用、可控体积、易演示、接口清晰（tokenize → encoder → softmax 取 argmax），且完全落在现有 `BaseModel` + ONNX Runtime 后端的能力内（新增输入 tokenizer 为纯 C++，复用上面 `tool::Tokenizer`）。标点恢复留作后续（YAGNI）。

### B.3.2 权重来源建议
- 情感/主题分类 BERT ONNX：从 HuggingFace / ModelScope 导出（选 mnli/情感微调 checkpoint → `optimum-cli export onnx`，或直接用社区已导出的 `bert-base-chinese`/`uer/roberta-base-finetuned-***` 分类 ONNX）。
- 放置：`test_data/test_models/onnx/`（如 `bert_sentiment.onnx`），与 ECAPA/DistilBERT 同约定位；版本/来源记录在 README 或 spec 附录。
- **测试/演示对缺失权重 SKIP 而非硬失败**（与 CV/Audio 真实权重路径一致）。

### B.3.3 现有代码适配点
- 复用 `BaseModel`（`ort::OrtBackend`）承载 ONNX 推理。
- 前端 tokenize（CLS/SEP、input_ids/attention_mask/token_type_ids）用 `tool::Tokenizer`（cppjieba 只做切词，需补 BPE/词表映射组件——新增一个轻量 tokenizer 或用模型自带词表）。
- postprocess：logits → softmax → top1 label + score。
- 文件：`csrc/nlp/solutions/text_classifier.h/.cpp`，`class TextClassifier : BaseModel`。

## B.4 NLP 六面交付
- **Python**：`csrc/pybind/nlp/`（`solutions_pybind.cpp` / `tools_pybind.cpp`）→ `nlp.solutions` / `nlp.tools`；按 `BUILD_NLP`（新开关，或并入 BUILD_AUDIO 依赖 cppjieba/wordpiece）门控。
- **CAPI**：工具层纯函数（`md_nlp_tokenize` / `md_nlp_split_sent` / `md_nlp_keywords` / `md_nlp_normalize` / `md_nlp_stats`）；分类方案 `md_nlp_classify(h, text, label*, score*)`。
- **C# / Rust**：薄封装随 CAPI；工具层函数式、易绑定。
- **demo + docs**：`examples/demo_nlp/`（分词/分句/关键词 + 可选情感分类）；README/EXAMPLES 加 NLP 能力行。工具层**无权重即可演示**；分类 demo 无权重 SKIP。

---

## 8. 测试
- `tests/test_solutions.cpp`（`[solutions]`）：每个方案用**合成轨迹/合成检测**做确定性断言（无权重）：
  - 计数：合成跨线轨迹 → 断言 in/out 计数。
  - 测速：合成等位移轨迹 + 时间戳 → 断言速度。
  - 热力图：合成轨迹 → 断言热度非零/峰值位置。
  - 裁剪/模糊：合成单框 → 断言尺寸/掩码。
  - 停车：合成框 vs 位元区域 → 断言占用状态。
  - 健身：合成肘角 → 断言动作计数。
- `tests/test_tools.cpp`（`[tools]`）：Detections 的 IoU/NMS/filter 确定性断言；Zone 跨线/区域计数；Metrics 小样本 mAP/P/R/F1；Slicer 切块大小；Smoother 收敛。
- 真实权重路径（若有视频/模型）可选，无权重 SKIP。

### 8.1 Audio 测试（`[audio_tools]` / `[audio_solution]`，合成音频即验证）
- `tests/test_audio_tools.cpp`（`[audio_tools]`，无权重）：
  - WavIO：写 → 读回，断言 meta/samples 一致；非 16k 也正确。
  - Resampler：正弦 8k→16k，断言输出长度与频率近似。
  - Mfcc/Fbank：纯正弦 → 断言 fbank 维度/非退化（合成特征可确定性断言）。
  - Waveform/Spectrum：合成单频正弦 → 断言频谱峰值落在该频率（可视化可校验关键 bin）。
  - VadSegment：用**合成 speech/silence 交替波形**（走真实 Silero 时需权重否则 SKIP；先按状态机/空段逻辑断言语义，权重缺失 SKIP）。
  - AudioMeta：构造已知 WAV 头 → 断言时长/采样率/位深。
- `tests/test_audio_solutions.cpp`（`[audio_solution]`）：
  - SpeakerDiarization：用合成两段不同音色波形（或 mock embedding 注入）+ 临界阈值 → 断言分段/speaker_id；无权重 SKIP。
  - StreamingSTT：合成波形 push → 断言 VAD 分段回调触发（转写文本依赖权重 SKIP）。
  - SpeakerSearch：复用 SpeakerGallery 语义（无权重可跑）——enroll 两个向量、match 断言正确 label。
  - TTSBatcher：enqueue 空/合并行为（不真的合成语音）或按权重 SKIP。

### 8.2 NLP 测试（`[nlp]`，合成文本即验证）
- `tests/test_nlp_tools.cpp`（`[nlp]`，无权重）：
  - Tokenizer：给定中文句 → 断言含预期词。
  - Splitter：含多种标点文本 → 断言句数/句边界。
  - Normalizer：中文数字/日期串 → 断言归一化结果。
  - Keywords：给定句 → 断言 top-K 含预期高频词。
  - Stats：断言字数/词数/句数。
- `tests/test_nlp_solutions.cpp`（`[nlp]`）：`TextClassifier` 构造/接口/错误路径（无权重跑）；真实分类结果依赖 `test_data/test_models/onnx/bert_sentiment.onnx`，缺失 SKIP。


## 9. 交付矩阵（CV / Audio / NLP 三表，各 6 面一次做齐）

### 9.1 CV

| 面 | 覆盖 |
|----|------|
| C++ 核心 | ✅（9 方案 + 6 工具） |
| Python | ✅（vision.solutions + vision.tools） |
| CAPI | ✅（solution 句柄） |
| C#/Rust | ✅（随 CAPI 薄封装） |
| demo+docs+tests | ✅（[solutions] + [tools]） |

### 9.2 Audio

| 面 | 覆盖 |
|----|------|
| C++ 核心 | ✅（4 方案 + 6 工具，`audio::solution`/`audio::tool`） |
| Python | ✅（audio.solutions + audio.tools） |
| CAPI | ✅（md_audio_* 方案句柄 + 工具函数） |
| C#/Rust | ✅（随 CAPI 薄封装） |
| demo+docs+tests | ✅（[audio_tools] + [audio_solution]） |

### 9.3 NLP

| 面 | 覆盖 |
|----|------|
| C++ 核心 | ✅（1 方案 TextClassifier + 5 工具，`nlp::solution`/`nlp::tool`） |
| Python | ✅（nlp.solutions + nlp.tools） |
| CAPI | ✅（md_nlp_tokenize/classify 等纯函数 + 方案句柄） |
| C#/Rust | ✅（随 CAPI 薄封装） |
| demo+docs+tests | ✅（[nlp]；分类方案依赖权重 → **权重外链、测试无权重 SKIP**） |

> 三域共用同一「方案层 + 工具层」理念与 6 面交付标准（用户已确认 Audio/NLP 与 CV 一致一次做齐）。NLP 方案层仅 TextClassifier 依赖小型 ONNX 权重（外链，仓库不含；测试/演示缺失 SKIP）。

## 10. 已知限制 / 假设
- 计数/测速精度依赖跟踪器稳定性（ByteTrack 已就位，作为限定）。
- 测速需合理时间戳（VideoDecoder 提供 pts_ms）+ 像素→米标定（用户提供 meter_per_pixel）。
- 健身计数基于肘/肩/腕角度阈值，为启发式（对齐 Ultralytics Workouts 简化版）。
- 停车需要用户定义位元区域。
- 工具层借鉴 supervision 语义，但为 C++ 原创实现（非移植 python 代码）。

### 10.1 Audio 已知限制
- 说话人日志精度依赖声纹（ECAPA embedding）+ VAD 分段质量：重叠说话、噪声、短语音段会导致错分/漏分。
- 聚类为阈值/聚合启发式，非重数无监督学习；人物数量需显式指定或按相似度阈值推断（YAGNI 不引入复杂聚类算法）。
- 实时转写非严格流式往返延迟保证（VAD 门控累积段长 + SenseVoice 转写耗时为上限）。
- VAD/声纹/SenseVoice 权重为外链（除仓库已有者外），测试对缺失 SKIP。

### 10.2 NLP 已知限制
- 项目**无 BERT/LLM 大模型地基**，因此 NLP 不做开放域对话 / 复杂 NLU / 生成式任务——范围收敛为「工具层 + 一个小型 ONNX 文本分类方案」。
- 标点恢复 / 主题分类等为**具体化可选**：默认只引入情感分类一个带权重方案，其余留待后续（YAGNI）。
- 关键词提取为启发式词频，无语义理解；分类精度受权重与语言（中/英）影响。

## 11. 成功标准
- 各方案/工具 C++ 类编译 + `[solutions]`/`[tools]` 合成确定性测试通过（无权重也可验证）。
- demo_solutions 综合演示（计数/热力图/测速叠加）可跑（真实视频或摄像头）。
- SDK 从「底层推理」升级为「有应用场景方案 + 可复用工具」，补齐与 Ultralytics Solutions / supervision 的对应关系。

### 11.1 Audio 成功标准
- Audio 各方案/工具 C++ 类编译 + `[audio_tools]`/`[audio_solution]` 合成确定性测试通过（无权重可验工具层与 SpeakerSearch 语义）。
- demo_diarization / demo_stream_stt（/ demo_tts_batch）可跑（真实 wav / 摄像头 / 文本队列）。

### 11.2 NLP 成功标准
- NLP 工具/方案 C++ 类编译 + `[nlp]` 合成文本测试通过（无权重也可验证全部工具层）。
- demo_nlp 可跑（分词/分句/关键词无权重即可演示）；TextClassifier demo 在有权重时跑通、无权重优雅 SKIP。

## 12. 对「SOTA 权重/真实测试」的安排
- 本 Item 聚焦应用层，不引入新 SOTA 权重；复用 yolo26n/姿态等已有真实权重点跑综合 demo。
- 其余待办（声纹 ecapa / ReID osnet / 手部 hand_pose / 动作 tsn/stgcn / 车辆 / 公式 / 文档真实权重下载与真测）单列一个「真实权重补全」后续项（可在本 Item demo 之后或并行进行，plan 里标注）。

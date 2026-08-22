# 声纹 / 说话人验证（Speaker Verification）——设计规范

- 日期：2026-08-22
- 状态：已批准（brainstorming）
- 路线：Item 4（顺序 4→5→7→6→8）
- 核心目标：为 ModelDeploy SDK 新增**说话人验证**能力——ECAPA-TDNN 说话人嵌入模型 + 内存说话人库（SpeakerGallery），6 面贯通（C++ / Python / CAPI / C# / Rust / demo+docs+tests），与本项目 Item 3（ReID）的「嵌入 + 库」模式对称。

---

## 1. 背景与动机

ModelDeploy 已具备完整 audio 模块（ASR 的 `SenseVoice`、TTS 的 `Kokoro`、VAD 的 `SileroVAD`），均为 `BaseModel` 子类，走 ORT/MNN 后端。视觉侧已有 ReID 的「512-d embedding + 内存 ReIdGallery（enroll/match top-k 余弦）」成熟模式。

本功能把该模式引入音频侧：说话人嵌入模型 + 说话人库，形成完整的说话人验证闭环（enroll 已知说话人 → 对新音频提取 embedding → 判定属于哪位说话人 / 是否陌生）。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope
- 独立说话人嵌入模型类 `SpeakerVerify`（ECAPA-TDNN，`.onnx`）。
- 内存说话人库 `SpeakerGallery`（enroll / remove / match / size / reset）。
- 6 面贯通：C++ 核心 / Python（pybind）/ CAPI / C# / Rust / demo + docs + tests。
- 音频前端：float PCM(16k) → mel-fbank（复用 kaldi-native-fbank）→ ECAPA 输入张量。
- `l2_normalize` 复用 `csrc/vision/utils.cpp:471`；相似度用 `compute_similarity`（对归一化向量 = 余弦）。

### Out-of-Scope（明确不做）
- 不做说话人**分割 / 聚类 / 日志**（仅验证 + 库匹配）。
- 不做流式说话人识别（仅整段 utterance 级 embed）。
- 不接入 ASR/VAD 自动切割（调用方自行提供干净 16k 音频段；VAD 可另行组合，本功能不内置）。
- 不做持久化 SpeakerGallery（内存态，与 ReIdGallery 一致）。
- 不做多说话人分离（说话人分离是另一模型/任务）。

## 3. 架构与组件

### 3.1 目录与命名（沿用 audio 约定）
```
csrc/audio/speaker_verify/
    ecapa.h / .cpp          # class SpeakerVerify : BaseModel
csrc/audio/speaker_gallery.h / .cpp   # class SpeakerGallery（内存库）
```
- 命名空间：`modeldeploy::audio::speaker_verify::SpeakerVerify`。
- `csrc/audio/*.cpp` 已被根 CMakeLists 的 `GLOB_RECURSE csrc/audio/*.cpp` 自动收集，无需改 CMake（探索已确认，行 117）。
- 新增 `.cpp` 是否依赖 audio 专用依赖：ecapa 仅用已捆绑的 `kaldi-native-fbank` + `csrc/utils`，无新第三方依赖。

### 3.2 `SpeakerVerify` 类（继承 `BaseModel`）
```cpp
namespace modeldeploy::audio::speaker_verify {
class MODELDEPLOY_CXX_EXPORT SpeakerVerify : public BaseModel {
public:
    SpeakerVerify(const std::string& model_file,
                  const float sample_rate = 16000.0f,
                  const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "SpeakerVerify"; }

    // 输入：float PCM 波形样本（16k）；输出：embedding（应用侧自行 l2 归一化）
    bool predict(const std::vector<float>& data, std::vector<float>* embedding);

    // 深拷贝：复用已加载 backend session（不重新加载模型）
    [[nodiscard]] std::unique_ptr<SpeakerVerify> clone() const;

    [[nodiscard]] bool is_initialized() const;

protected:
    bool initialize();
    bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* embedding);

private:
    explicit SpeakerVerify() = default;  // 供 clone()
    int32_t mel_bins_{80};               // ECAPA 固定 80 维 mel
    float sample_rate_{16000.0f};
};
} // namespace modeldeploy::audio::speaker_verify
```

### 3.3 `SpeakerGallery`（复用 ReIdGallery 模式）
```cpp
namespace modeldeploy::audio {
class MODELDEPLOY_CXX_EXPORT SpeakerGallery {
public:
    SpeakerGallery() = default;
    void enroll(const std::string& label, const std::vector<float>& embedding);
    void remove(const std::string& label);
    std::vector<std::pair<std::string, float>>
        match(const std::vector<float>& embedding, size_t top_k) const;
    size_t size() const;
    void reset();

private:
    std::map<std::string, std::vector<float>> speakers_;  // label -> l2 normalized embedding
};
}
```
- enroll 时对 embedding 做 `l2_normalize` 后存储；match 时 query 先 `l2_normalize`，再用 `compute_similarity`（余弦）。
- 同 label 重复 enroll = 覆盖。
- match 返回 `(label, score)`，按 score 降序，取 top_k。

### 3.4 数据流
```
float PCM(16k)
   └─ preprocess: 分帧(25ms/10ms) → mel-fbank(80) → Tensor [1,80,T]
        └─ ECAPA infer → Tensor [1,D] (D=192 或 512)
             └─ postprocess: 取 [0][:] → std::vector<float> embedding
SpeakerGallery.enroll/math: l2_normalize → compute_similarity → top_k
```

## 4. Python（pybind）
- 新增 `csrc/pybind/audio/speaker_verify_pybind.cpp`，定义 `bind_speaker_verify(pybind11::module&)`：
  - `SpeakerVerify(model_file)`：构造；`.predict(samples: List[float]) -> List[float]`（embedding）；`.is_initialized()`。
  - `SpeakerGallery`：`.enroll(label, emb)` / `.remove(label)` / `.match(emb, top_k=1) -> List[Tuple[str,float]]` / `.size()` / `.reset()`。
- 注册入口 `csrc/pybind/main.cpp`：在 `#ifdef BUILD_AUDIO` 下 `audio::bind_speaker_verify(audio_module)`（探索已确认 main.cpp 行 8-42 结构）。
- 注意：根 CMakeLists 行 310-312 在 `NOT BUILD_AUDIO` 时要一并 `REMOVE_ITEM` 新增的 `speaker_verify_pybind.cpp`（与 kokoro_pybind 相同处理）。

## 5. CAPI
- 枚举：`capi/md_capi.h` `MD_MODEL_KIND`（行 96-97 附近）新增 `MD_MODEL_SPEAKER_VERIFY`（在 `MD_MODEL_COUNT` 前；注意与 Item 2/3 的 HAND=27/REID=28 对齐现状，新值从 29 起或按当前 count 追加）。
- `md_model_create` switch 的 `#ifdef BUILD_AUDIO` 块（行 928-948）：新增构造 `new audio::speaker_verify::SpeakerVerify(parts[0], opt)`。
- `md_model_handle::~md_model_handle`（行 999-1002）：新增 `delete` 分发。
- 新推理入口 `md_audio_speaker_embed(h, const float* samples, size_t n, const float** embedding, size_t* emb_n)`：
  - 返回**借用指针**（指向内部 vector），生命周期由 ResultHandle 管控 —— 与 `md_result_reid_embedding` 同语义。
  - 或返回写入调用方 buffer：`md_audio_speaker_embed(h, samples, n, float* out_emb, size_t* out_n)`（调用方分配）。**采用借用指针语义**（与现有 audio CAPI 的 `const float**` 一致）。
- 库相关 CAPI 可选：`md_speaker_gallery_*`。YAGNI 收紧——本计划中 SpeakerGallery 的 CAPI 面**不做**（库匹配在 Python/C++ 层足够；避免 CAPI 面扩大）。若后续需要再补。→ **决策：SpeakerGallery 仅 C++/Python/Rust（可选），CAPI 只做单模型 embed。** 见 §10 交付矩阵。

## 6. C#
- `csharp/ModelDeploy/` 新增 `SpeakerVerifyModel`（仿 `Kokoro`/`ReIdModel` 模式）：
  - 构造：`SpeakerVerifyModel(string modelPath)` → 内部调 `md_model_create(MD_MODEL_SPEAKER_VERIFY, ...)`。
  - `float[] Predict(float[] samples)`：调 `md_audio_speaker_embed`，`Marshal.Copy` 立即复制 embedding（借用指针 → 托管数组，与 `ReadFloats` 模式一致）。
  - `MDModelKind` 枚举（`types_internal_c.cs`）新增 `MD_MODEL_SPEAKER_VERIFY`（值对齐 CAPI）。
  - 测试 `SpeakerVerify_Works`：无权重时 Skipped；仅验证构造/枚举/错误路径。

## 7. Rust
- `rust/modeldeploy/` 新增 `SpeakerVerify`（仿 `ReID` 的 `model_wrapper!`）：
  - `ffi.rs` `MDModelKind` 新增 `SpeakerVerify`（值对齐 29）、`extern "C"` 声明 `md_audio_speaker_embed`。
  - `model.rs` `SpeakerVerify::new(model, opt)` + `predict(&[f32]) -> Vec<f32>`；embedding `read_f32` 复制（安全）。
  - 测试 `test_speaker_verify`：无权重跳过。

## 8. demo + docs
- `examples/demo_speaker/`：`demo_speaker.cpp + CMakeLists.txt`（单 add_executable + 链 `${LIBRARY_NAME}`；audio 不依赖 OpenCV）。
  - 用法：入参模型路径 + 两个 wav，各提取 embedding，enroll 一个、match 另一个 → 打印 `(label, score)`。
  - 无权重时打印错误不崩溃。
- `examples/CMakeLists.txt` 加 `add_subdirectory(demo_speaker)`；`examples/EXAMPLES.md` 加行（3 列）。
- README 能力列表（第 3 行附近）加"声纹/说话人验证"。

## 9. 测试
- `tests/test_speaker_verify.cpp`（Catch2 标签 `[speaker]`）：
  - Test A（无权重可跑）：SpeakerGallery 语义——enroll 两个向量、match 返回正确 label/降序 score、remove/size/reset。
  - Test B（需 `test_data/test_models/onnx/ecapa_tdnn.onnx`，缺失 SKIP）：构造 SpeakerVerify → 对合成波形预测 → embedding 维度断言（192 或按实际模型）。
- `tests/CMakeLists.txt` 的 `TEST_SOURCES` 加 `test_speaker_verify.cpp`。
- Rust 集成测试 `test_speaker_verify`、C# `SpeakerVerify_Works`（缺失权重 SKIP）。
- CAPI 测试：`test_capi.cpp` 新增 speaker 用例（`[capi]` 标签；模型缺失 guard 跳过）。

## 10. 交付矩阵（6 面）

| 面 | 内容 | 完成标准 |
|----|------|---------|
| C++ 核心 | `SpeakerVerify`(BaseModel) + `SpeakerGallery` | 编译 + Test A 通过 |
| Python | `speaker_verify_pybind` + main 注册 | import + predict + gallery 冒烟 |
| CAPI | `MD_MODEL_SPEAKER_VERIFY` + `md_audio_speaker_embed` + 分发 | 编译 + capi 用例 |
| C# | `SpeakerVerifyModel` + 枚举 | 编译 + `SpeakerVerify_Works` |
| Rust | `SpeakerVerify` + ffi | clippy 干净 + 测试 |
| demo+docs+tests | demo_speaker + README/EXAMPLES + test_speaker | 编译 + Usage 路径 |

> 说明：SpeakerGallery 的 CAPI/C# 面按 YAGNI 收紧——CAPI 仅暴露单模型 embed；库匹配由 Python/C++/Rust 使用。这满足"说话人验证"主需求而不过度扩大。

## 11. 已知限制 / 假设
- ECAPA-TDNN ONNX 权重不在仓库 / 官方 test_data 内；需外部下载放置到 `test_data/test_models/onnx/ecapa_tdnn.onnx`（测试对缺失 SKIP 而非硬失败）。
- ECAPA 输入：80 维 mel，16k；若采用不同输入规格的 ONNX（如 40 维 mel 或 8k），preprocess 的 mel_bins/sample_rate 需对齐实际模型。**默认按 80 mel / 16k**，实现时以模型实际 input 形状为准（运行时按 `get_input_info(0)` 探测 T 维；mel_bins 固定 80）。
- embedding 维度跟随模型（192 常见；若导出为 512 亦支持，由输入 shape 决定）。
- 排序与 top_k 语义与 ReIdGallery 一致（升序敏感测试按实际实现断言）。

## 12. 风险
- ECAPA ONNX 适配：不同导出的输入 layout（`[1,80,T]` vs `[T,80]`）可能不同。缓解：运行时读 `get_input_info(0)` 的 4D shape，按第 1 维/第 2 维动态分派，或要求固定 `[N,80,T]` 并在 spec 固定。
- 音频前端与 ASR 一致性：mel 参数（窗长/移位）与 ECAPA 训练设置需一致，否则 embedding 质量差；实现以模型配套的预处理（均值/方差归一化可选）为准。

## 13. 成功标准
- C++/Python/CAPI/C#/Rust 均编译通过；无权重时优雅 SKIP。
- SpeakerGallery Test A 恒定通过（不依赖权重）。
- demo_speaker 编译 + Usage/错误路径验证。
- 与 Item 3 ReID 对称，后续替换更强声纹模型（如 wav2vec2）只需换 ONNX 文件。

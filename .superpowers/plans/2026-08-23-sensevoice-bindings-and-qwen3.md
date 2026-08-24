# SenseVoiceResult 外出绑定 + 本地 ITN 评估 + Qwen3-ASR 移植 计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成四项：①本地 ITN 能力评估（是否需第三方库）；②SenseVoice 结构化结果 (`SenseVoiceResult`) 经 capi/C#/Rust 外出；③Qwen3-ASR 移植接入 SDK；④Qwen3-TTS 移植（当前被封锁，见 Task 5）。

**Architecture:** SenseVoiceResult 已在 C++ 端（`csrc/audio/asr/sense_voice.{h,cpp}`）结构化。外出路径分为三层：C API 层新增结果结构体+两个函数并把结果暂存到模型句柄、Rust/C# 通过 FFI 调 capi。Qwen3-ASR 以 `HeiSir2014/sherpa-onnx` 分支（上游已合并）的 `offline-qwen3-asr-*` 为基准，按 ModelDeploy 的 ORT 后端移植 LLM 自回归解码循环。

**Tech Stack:** C++17, ORT(ONNX Runtime), pybind11, C API, C# P/Invoke, Rust

## Global Constraints
- 全部走 C++17；MSVC 需 `/utf-8`（根 CMakeLists 已设置）。
- 语言文本含中文：源码保持 UTF-8，C#/Rust 字符串经 UTF-8 bytes 传递（`uint8_t*`/`char*`）。
- 不新增第三方运行时依赖到 SDK 核心；ITN 全覆盖用可选 `ENABLE_WETEXT` 后端（`WETEXT_INCLUDE_DIR`）。
- capi 结果字符串“归句柄所有，零拷贝借用”，生命周期与 `md_model_destroy`/下次 predict 一致（沿用 `text_buf` 模式）。
- 测试数据（Qwen3-ASR 权重 ~916MB）不入库；经 `wget/curl -C -` 断点续传。
- 提交粒度：每个可独立验证的交付物一次提交；不 push 除非用户要求。

---

## Task 0: 本地 ITN 能力评估与结论（已完成侦察，落档）

**Files:**
- Read: `csrc/audio/tools/itn.h`, `csrc/audio/tools/itn.cpp`, `tests/test_audio_tools.cpp`
- Result: 写入本 plan 的“结论”区（不新增代码）

**结论（实测，C++ 48 断言 + python 实测）:**
- 已支持：纯数字（123/3004）、小数（3.14）、百分数（5%）、四位年份（2024年）、日期（5月9日）、时间（5:30/9:05）、序数（第9）、含万/亿大数（12亿7200万）、小数+折（3.5折）、货币（320元）、负数（负15）。
- 覆盖不足：真分数（二分之一→原样）、量词误切（**“五十千克”→“1050克”，应为“50千克”**，实为 bug）、“九月”不转“9月”、电话号码分段。
- **结论**：轻量实现作为默认零依赖核心够用；**是否需第三方库取决于覆盖要求**——通用口语数字/日期/时间无需；若要生产级全量（分数/量词/电话/货币语义），应启用可选 `ENABLE_WETEXT` 后端（`ItnEngine(ItnBackend::WeText)`，需 OpenFst + `WETEXT_INCLUDE_DIR`）。**@用户决定**：是否需要修“量词误切”这个轻量 bug，还是直接以 WeText 后端兜底（推荐兜底）。

---

## Task 1: C API 外出 SenseVoiceResult

**Files:**
- Modify: `capi/md_capi.h`（新增 `MDAsrResult` 与两个 `md_audio_asr*_result` 函数声明）
- Modify: `capi/md_capi.cpp`（`md_model_handle` 加暂存字段；实现两函数）
- Test: `capi/test_capi_audio.c`（扩展 run_asr 校验结构化字段）

**Interfaces:**
- Consumes: `csrc/audio/asr/sense_voice.h` 的 `SenseVoice::predict(data,&SenseVoiceResult)` 与 `SenseVoiceResult{text,language,emotion,event,task,itn,nospeech}`。
- Produces（供 Task 2/3 使用）:
```c
typedef struct MDAsrResult {
    const char* text;
    const char* language;
    const char* emotion;
    const char* event;
    const char* task;
    int itn;
    int nospeech;
} MDAsrResult;
MD_CAPI_EXPORT MDStatus md_audio_asr(MDModelHandle h, const float* samples, size_t n,
                                     int sample_rate, const char** text);
MD_CAPI_EXPORT MDStatus md_audio_asr_result(MDModelHandle h, const float* samples, size_t n,
                                            int sample_rate, MDAsrResult* out);
MD_CAPI_EXPORT MDStatus md_audio_asr_wav_result(MDModelHandle h, const char* wav_path, MDAsrResult* out);
```

- [ ] **Step 1: 在 `md_capi.h` 的音频小节（`md_audio_asr_wav`/`md_audio_asr` 附近）声明 `MDAsrResult` 与两个 `*_result` 函数**
- [ ] **Step 2: 在 `md_capi.cpp` 的 `struct md_model_handle`（行 96-103）追加暂存字段**
```cpp
std::string asr_text_, asr_lang_, asr_emotion_, asr_event_, asr_task_;
bool asr_itn_ = false, asr_nospeech_ = false;
```
- [ ] **Step 3: 实现 `md_audio_asr_result`/`md_audio_asr_wav_result`**（在 `md_audio_asr` 附近）：新建私有静态 helper `fill_asr_result(SenseVoice&, const std::vector<float>&, MDAsrResult&)`，调用 `predict(&r)` 后把六个字符串字段拷入句柄暂存、`out` 指针指向暂存 `c_str()`。镜像现有 `md_audio_asr` 的空指针/`is_initialized`/`set_error` 校验。
- [ ] **Step 4: 扩展 `test_capi_audio.c` run_asr**：调用 `md_audio_asr_wav_result`，校验 `out->language=="zh"`（对 zh.wav）、`out->text` 非空。
- [ ] **Step 5: 编译并跑 capi 测试**
```bash
# build 目录需 BUILD_CAPI=ON；当前 build 未知，用 build_py 不含 capi → 用 build 重新配置或 build_tdc
cd /mnt/e/CLionProjects/ModelDeploy && /usr/bin/cmake -S . -B build -DBUILD_CAPI=ON >/dev/null 2>&1
/usr/bin/cmake --build build --config Release --target md_capi -j4
# 或直接 ctest/capi 目标；用 build_capi 单独 target 最稳
```
Expected: 无编译错误，跑通结构化 ASR。
- [ ] **Step 6: Commit** `git add capi/md_capi.h capi/md_capi.cpp capi/test_capi_audio.c && git commit -m "capi: expose SenseVoiceResult structured ASR"`

---

## Task 2: C# 外出 SenseVoiceResult

**Files:**
- Modify: `csharp/ModelDeploy/NativeMethods.cs`（DllImport 声明）
- Modify: `csharp/ModelDeploy/AudioModels.cs`（`SenseVoice` 封装 + 结果 DTO）
- Test: `csharp/ModelDeployUnitTest/CapiAudioTests.cs`

**Interfaces:**
- Consumes: Task 1 的 `md_audio_asr_wav_result` 与 `MDAsrResult`（C 结构体布局：6 个指针 + 2 个 int，需按 `[StructLayout(LayoutKind.Sequential)]` / `CharSet.Ansi` 定义）。
- Produces: `SenseVoice` 类增加 `RecognizeResult RecognizeFromWav(string path)` / `MdAsrResult` DTO。

- [ ] **Step 1: NativeMethods.cs 增 P/Invoke**
```csharp
[StructLayout(LayoutKind.Sequential)]
public struct MdAsrResult {
    public IntPtr text; public IntPtr language; public IntPtr emotion;
    public IntPtr event_; public IntPtr task; public int itn; public int nospeech;
}
[DllImport(NativeLib, CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_audio_asr_wav_result(IntPtr model, string wavPath, out MdAsrResult outResult);
```
- [ ] **Step 2: AudioModels.cs 的 SenseVoice 增加 `RecognizeFromWav`**：调用 native，把各 IntPtr `Marshal.PtrToStringUTF8` 解出，返回 DTO。
- [ ] **Step 3: 补单测**（对 `test_data/test_models/onnx/sense_voice/test_wavs/zh.wav` 断言 language=="zh"）。
- [ ] **Step 4: 编译运行**：`dotnet build csharp/ModelDeploy` + `dotnet test`。
- [ ] **Step 5: Commit**

---

## Task 3: Rust 外出 SenseVoiceResult

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（新增 extern "C" 声明）
- Modify: `rust/modeldeploy/src/audio.rs`（新增 `SenseVoice::recognize_wav` 返回结构化）
- Test: `rust/modeldeploy/tests/integration_test.rs`

**Interfaces:**
- Consumes: Task 1 的 `md_audio_asr_wav_result`/`MDAsrResult`。
- Produces: `pub struct AsrResult { text, language, emotion, event, task: String, itn, nospeech: bool }`，`SenseVoice::recognize_wav(path)->Result<AsrResult,MdError>`。

- [ ] **Step 1: ffi.rs 加绑定**
```rust
#[repr(C)] pub struct MDAsrResult {
    pub text: *const libc::c_char, pub language: *const libc::c_char,
    pub emotion: *const libc::c_char, pub event: *const libc::c_char,
    pub task: *const libc::c_char, pub itn: i32, pub nospeech: i32,
}
pub fn md_audio_asr_wav_result(model: ffi::MDModelHandle, wav_path: *const libc::c_char,
                               out: *mut MDAsrResult) -> MDStatus;
```
- [ ] **Step 2: audio.rs 新增 `AsrResult` + `SenseVoice`（含 `new`/`recognize`/`recognize_wav`）**，将各指针 `CStr::from_ptr` 转 String，`itn/nospeech` 由 `i32!=0` 转 bool。
- [ ] **Step 3: 集成测试**：对 zh.wav 断言 `language=="zh"`。
- [ ] **Step 4: 编译运行**：`cargo test --manifest-path rust/modeldeploy/Cargo.toml`（或按仓库现有测试命令）。
- [ ] **Step 5: Commit**

---

---

# 补充任务（用户新需求 2026-08-23）：流式 ASR + ASR 最佳解决方案

**目标：** SDK 补一个真正的流式 ASR（Paraformer-streaming），并做“ASR 最佳方案”：实时流式出部分结果，VAD 判句结束后按置信度触发离线 SenseVoice 精修。

**模型选型（用户已确认）：** Paraformer-streaming（中英双语，非 AR，无解码环，接近 SenseVoice 复杂度）。
**精修真策略（用户已确认）：按置信度**——流式结果置信度足够高才保留流式，否则跑离线才替换。

**权重下载（用户侧，modelscope/HF 二选一）：**
- modelscope: `csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en`（命令 `modelscope download --model ... --local_dir ...` 或 tar.bz2）
- HF: `https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/main/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2`
- 解压目录含 `encoder.onnx`、`decoder.onnx`、`tokens.txt`

**算法契约（已从 sherpa `online-recognizer-paraformer-impl` 研读确认，参考在 /tmp/ps/）：**
- 流式特征：`knf::OnlineFbank`（SDK 已链 kaldi-native-fbank；paraformer 特征设为 `normalize_samples=false`、hamming、`snip_edges=true`、`feature_dim=80` 梅尔）。
- 常量：`chunk_size_=61`、`left_chunk_size_=5`、`right_chunk_size_=3`；LFR 窗口/移位来自模型元数据（默认 window=7, shift=6）。
- encoder 输入：`[1, num_frames(=LFR 后 10), feat_dim=80*7=560]` float + `int32 len`；输出[0]=encoder_out、[1]=encoder_out_len、[2]=alpha（CIF alpha）。
- 元数据：`vocab_size`、`lfr_window_size`、`lfr_window_shift`、`encoder_output_size`、`decoder_num_blocks`、`decoder_kernel_size`、`neg_mean`(cmvn)、`inv_stddev`(cmvn，初始化后 ×√encoder_output_size)。
- CIF 搜索：积分 alpha、阈值 1.0、fire 时生成 acoustic_embedding（累积权重）；decoder 输入 `encoder_out, len, acoustic_embedding, len, states...`；输出[1]=sample_ids(int64)、[2..]=新 states。
- 解码：tokens.txt（SymbolTable）→ 文本（`@@` 处理：token 以 `@@` 结尾表示与下一段合并成词）。

**Todo 清单见会话 todowrite。**

---

## Task 4: Qwen3-ASR 移植 — 取消
**状态：** 用户决策改由 **vLLM-omni** 部署，SDK 保持轻量，不再移植。权重已下载解压于 `test_data/test_models/onnx/qwen3_asr/`，非 SDK 所需，可自行清理。

## Task 5: Qwen3-TTS 移植 — 取消
**状态：** 同 Task 4，改由 vLLM-omni 部署。ONNX 权重无官方发布，源为 `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`（PyTorch，需 `scripts/qwen3-tts/export-onnx.py` 导出 9 个子模型）。不再移植。

---

## Self-Review
- **覆盖**：①Task 0 评估✅；②Task 1-3 三语言外出✅；③Task 4 移植✅（含下载门禁）；④Task 5 标注封闭，无来源不硬做✅。
- **占位符**：Task 0/5 是“需用户决策/被封锁”的显式说明，非占位实现。
- **类型一致**：`MDAsrResult` 字段在 Task 1 定义、Task 2(C# `MdAsrResult`)/3(Rust `MDAsrResult`) 按同一布局消费，字段名/顺序一致。

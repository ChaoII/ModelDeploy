# Item 4: 声纹 / 说话人验证（Speaker Verification）——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 ModelDeploy SDK 新增说话人嵌入模型 `SpeakerVerify`（ECAPA-TDNN C512，SOTA+轻量）与内存说话人库 `SpeakerGallery`，6 面贯通（C++/Python/CAPI/C#/Rust/demo+docs+tests）。

**Architecture:** `SpeakerVerify` 继承 `BaseModel`，float PCM(16k) → 80 维 mel-fbank（复用 kaldi-native-fbank）→ ECAPA 推理 → 192-d embedding；`SpeakerGallery` 复用 `ReIdGallery` 模式（l2 归一化 + `compute_similarity` 余弦 top-k）。CAPI 仅暴露单模型 embed（SpeakerGallery 的 CAPI/C# 面按 YAGNI 不做）。

**Tech Stack:** C++17、pybind11、kaldi-native-fbank、ECAPA-TDNN ONNX、Catch2（测试）、现有 `BaseModel`/`Runtime` 抽象。

**Spec:** `docs/superpowers/specs/2026-08-22-speaker-verification-design.md`

## Global Constraints

- C++17 必需；MSVC 必须 `/utf-8`（根 CMake 已为 SDK 自动设置）。
- 新增 `.cpp` 放 `csrc/audio/` 下会自动被 `GLOB_RECURSE csrc/audio/*.cpp` 收集，无需改 CMake 主列表。
- 3 面前端交互（回测）：无 ECAPA 权重时测试 SKIP 而非硬失败（`test_data/test_models/onnx/ecapa_tdnn.onnx` 缺失 → SKIP）。
- 跨后端语义：`SpeakerVerify` 继承 `BaseModel` → 天然 ORT/MNN/TRT/Sophgo；不得在 speaker 源码引入特定后端 include。
- 命名空间：C++ `modeldeploy::audio::speaker_verify::SpeakerVerify`、`modeldeploy::audio::SpeakerGallery`。
- SpeakerGallery 面范围：C++/Python/Rust ✅；CAPI/C# ❌（YAGNI）。
- 复用 `csrc/vision/utils.h` 的 `l2_normalize(values)` 与 `compute_similarity(f1,f2)`（`std::vector<float>`，audio 可直接用）。
- ECAPA 输入固定 80 维 mel，16k；embedding 维度跟随模型（默认 192-d），由 `get_input_info(0)`/`get_output_info(0)` 在 initialize 时探测。
- 构建命令（Windows）：`.bat` 包裹 `vcvars64.bat` 经 `cmd /c`；`cmake --build <build_dir> --config Release --parallel 8`。

---

### Task 1: `SpeakerVerify` C++ 核心类 + `SpeakerGallery`

**Files:**
- Create: `csrc/audio/speaker_verify/ecapa.h`
- Create: `csrc/audio/speaker_verify/ecapa.cpp`
- Create: `csrc/audio/speaker_gallery.h`
- Create: `csrc/audio/speaker_gallery.cpp`
- Test: `tests/test_speaker_verify.cpp`

**Interfaces:**
- Produces (later tasks rely on):
  - `modeldeploy::audio::speaker_verify::SpeakerVerify(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption())`
  - `bool SpeakerVerify::predict(const std::vector<float>& data, std::vector<float>* embedding)`
  - `std::string SpeakerVerify::name() const` → `"SpeakerVerify"`
  - `std::unique_ptr<SpeakerVerify> SpeakerVerify::clone() const`
  - `bool SpeakerVerify::is_initialized() const`
  - `modeldeploy::audio::SpeakerGallery::enroll(label, embedding)`
  - `std::vector<bool> SpeakerGallery::remove(label)`
  - `std::vector<std::pair<std::string,float>> SpeakerGallery::match(embedding, k)`
  - `size_t SpeakerGallery::size()`
  - `void SpeakerGallery::clear()`

- [ ] **Step 1: Write the failing tests (Test A: gallery without weights)**

`tests/test_speaker_verify.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "csrc/audio/speaker_gallery.h"

using namespace modeldeploy::audio;

TEST_CASE("SpeakerGallery enroll/match/remove", "[speaker]") {
    SpeakerGallery g;
    g.enroll("alice", std::vector<float>{1.0f, 0.0f, 0.0f});
    g.enroll("bob", std::vector<float>{0.0f, 1.0f, 0.0f});
    REQUIRE(g.size() == 2);
    auto top = g.match(std::vector<float>{0.99f, 0.01f, 0.0f}, 1);
    REQUIRE(top.size() == 1);
    REQUIRE(top[0].first == "alice");
    REQUIRE(top[0].second > 0.9f);
    g.remove("alice");
    REQUIRE(g.size() == 1);
    g.clear();
    REQUIRE(g.size() == 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build_tdc_gpu && .\bin\test_modeldeploy.exe "[speaker]"`
Expected: FAIL（编译失败或 test case 未定义——speaker_gallery.h 不存在）

- [ ] **Step 3: Write minimal implementation — `csrc/audio/speaker_gallery.h`**

```cpp
//
// Created for in-memory speaker feature gallery.
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/md_decl.h"
#include "vision/utils.h"

namespace modeldeploy::audio {
    /*! @brief In-memory speaker gallery: label -> l2-normalized embedding, top-k cosine match.
     *  match 对 query 先 l2_normalize 再 compute_similarity（=余弦）。
     */
    class MODELDEPLOY_CXX_EXPORT SpeakerGallery {
    public:
        void clear() { gallery_.clear(); }
        void enroll(const std::string& label, const std::vector<float>& embedding);
        std::vector<bool> remove(const std::string& label);
        std::vector<std::pair<std::string, float>> match(const std::vector<float>& embedding, int k) const;
        size_t size() const { return gallery_.size(); }
    private:
        std::map<std::string, std::vector<float>> gallery_;
    };
} // namespace modeldeploy::audio
```

- [ ] **Step 4: Write `csrc/audio/speaker_gallery.cpp`**

```cpp
#include "csrc/audio/speaker_gallery.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::audio {
    void SpeakerGallery::enroll(const std::string& label, const std::vector<float>& embedding) {
        gallery_[label] = vision::l2_normalize(embedding);   // 覆盖同 label
    }

    std::vector<bool> SpeakerGallery::remove(const std::string& label) {
        std::vector<bool> ok{ false };
        auto it = gallery_.find(label);
        if (it != gallery_.end()) {
            gallery_.erase(it);
            ok[0] = true;
        }
        return ok;
    }

    std::vector<std::pair<std::string, float>>
    SpeakerGallery::match(const std::vector<float>& embedding, int k) const {
        const auto q = vision::l2_normalize(embedding);
        std::vector<std::pair<std::string, float>> scored;
        scored.reserve(gallery_.size());
        for (const auto& [label, ref] : gallery_) {
            scored.emplace_back(label, vision::compute_similarity(q, ref));
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        if (scored.size() > static_cast<size_t>(k)) scored.resize(k);
        return scored;
    }
} // namespace modeldeploy::audio
```

- [ ] **Step 5: Write `csrc/audio/speaker_verify/ecapa.h`**

```cpp
#pragma once

#include "csrc/base_model.h"

namespace modeldeploy::audio::speaker_verify {
    /*! @brief ECAPA-TDNN speaker embedding model (SOTA + lightweight).
     *  输入 float PCM(16k)，输出 192-d 说话人 embedding。
     */
    class MODELDEPLOY_CXX_EXPORT SpeakerVerify : public BaseModel {
    public:
        SpeakerVerify(const std::string& model_file,
                      const RuntimeOption& custom_option = RuntimeOption());
        [[nodiscard]] std::string name() const override { return "SpeakerVerify"; }
        // 输入 float PCM(16k)；输出 embedding（len 由模型决定，默认 192）
        bool predict(const std::vector<float>& data, std::vector<float>* embedding);
        [[nodiscard]] bool is_initialized() const;
        [[nodiscard]] std::unique_ptr<SpeakerVerify> clone() const;
    protected:
        bool initialize();
        bool preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs);
        bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* embedding);
    private:
        explicit SpeakerVerify() = default;   // 供 clone()
        int32_t mel_bins_{80};
        int32_t embedding_dim_{-1};
        std::vector<float> window_;  // 备用（无需时保持空）
    };
} // namespace modeldeploy::audio::speaker_verify
```

- [ ] **Step 6: Write `csrc/audio/speaker_verify/ecapa.cpp`**

```cpp
#include "csrc/audio/speaker_verify/ecapa.h"
#include <algorithm>
#include <csrc/utils/utils.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy::audio::speaker_verify {
    SpeakerVerify::SpeakerVerify(const std::string& model_file,
                                 const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    std::unique_ptr<SpeakerVerify> SpeakerVerify::clone() const {
        auto clone_model = std::unique_ptr<SpeakerVerify>(new SpeakerVerify());
        clone_model->set_runtime(const_cast<SpeakerVerify*>(this)->clone_runtime());
        clone_model->runtime_option = runtime_option;
        clone_model->mel_bins_ = mel_bins_;
        clone_model->embedding_dim_ = embedding_dim_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

    bool SpeakerVerify::is_initialized() const { return initialized_; }

    bool SpeakerVerify::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "SpeakerVerify: failed to init runtime." << std::endl;
            return false;
        }
        // 探测 embedding 维度（输出张量最后一维）
        if (num_outputs() > 0) {
            auto out_info = get_output_info(0);
            const auto& shape = out_info.shape;
            if (!shape.empty()) embedding_dim_ = static_cast<int32_t>(shape.back());
        }
        if (embedding_dim_ <= 0) {
            MD_LOG_WARNING << "SpeakerVerify: could not detect embedding dim, default 192." << std::endl;
            embedding_dim_ = 192;
        }
        return true;
    }

    bool SpeakerVerify::preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs) {
        if (data.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: input data is empty." << std::endl;
            return false;
        }
        knf::FbankOptions opts;
        opts.frame_opts.dither = 0;
        opts.frame_opts.snip_edges = false;
        opts.frame_opts.window_type = "hamming";
        opts.frame_opts.samp_freq = 16000;
        opts.mel_opts.num_bins = mel_bins_;
        knf::OnlineFbank kaldi_f_bank(opts);
        kaldi_f_bank.AcceptWaveform(16000, data.data(), static_cast<int32_t>(data.size()));
        kaldi_f_bank.InputFinished();
        const int32_t n = kaldi_f_bank.NumFramesReady();
        std::vector<float> feats;
        feats.reserve(static_cast<size_t>(n) * mel_bins_);
        for (int32_t i = 0; i < n; ++i) {
            const auto* frame = kaldi_f_bank.GetFrame(i);
            for (int32_t k = 0; k < mel_bins_; ++k) feats.push_back(frame[k]);
        }
        if (feats.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: no fbank frames produced." << std::endl;
            return false;
        }
        // ECAPA 输入 [1, mel_bins, T]（以模型实际 shape 为准；若为 [1, T, mel_bins] 则交换）
        const int64_t T = static_cast<int64_t>(n);
        if (T == 0) return false;
        // 缺省列为 [1,80,T]，实现时按 get_input_info(0).shape 对齐（此处用标准 ECAPA 布局）
        const std::vector<int64_t> shape = {1, mel_bins_, T};
        std::vector<float> transposed(feats.size());
        // frame-major( T x mel ) -> channel-major( mel x T )
        for (int64_t t = 0; t < T; ++t)
            for (int32_t k = 0; k < mel_bins_; ++k)
                transposed[k * T + t] = feats[t * mel_bins_ + k];
        outputs->resize(1);
        (*outputs)[0] = std::move(Tensor(transposed.data(), shape, DataType::FP32, Device::CPU));
        return true;
    }

    bool SpeakerVerify::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* embedding) {
        if (infer_result.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: no inference result." << std::endl;
            return false;
        }
        auto& t = infer_result[0];
        const auto* p = static_cast<const float*>(t.data());
        const int64_t stride = t.size();
        for (int64_t i = 0; i < stride; ++i) embedding->push_back(p[i]);
        // 模型输出通常已归一化；若需 L2 再归一化由应用/库层做，这里原样返回。
        return true;
    }

    bool SpeakerVerify::predict(const std::vector<float>& data, std::vector<float>* embedding) {
        if (!preprocess(data, &reused_input_tensors_)) return false;
        for (int i = 0; i < reused_input_tensors_.size(); i++)
            reused_input_tensors_[i].set_name(get_input_info(i).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "SpeakerVerify: inference failed." << std::endl;
            return false;
        }
        if (!postprocess(reused_output_tensors_, embedding)) return false;
        return true;
    }
} // namespace modeldeploy::audio::speaker_verify
```

> **实现提示**：ECAPA-TDNN 官方 ONNX 的输入布局可能是 `[1,80,T]` 或 `[1,T,80]`。**Task 1 按 `[1,80,T]`（channel-first）实现**；若实际模型为另一布局，在集成回归（Task 8）时按 `get_input_info(0).shape` 校正 preprocess 的转置逻辑并更新注释。这个差异在 Task 1 不作为失败点（权重外链），仅保持代码按标准 ECAPA 布局。

- [ ] **Step 7: Run tests to verify pass**

Run: `cd build_tdc_gpu && .\bin\test_modeldeploy.exe "[speaker]"`
Expected: PASS（Test A gallery 恒定通过；SpeakerVerify 构造无权重路径不测 or 只需编译）

- [ ] **Step 8: Verify compile (SDK)**

Run: `.bat` 构建 `build_tdc_gpu`
Expected: 0 errors（`ecapa.cpp`/`speaker_gallery.cpp` 被 GLOB 收集编译）

- [ ] **Step 9: Commit**

```bash
git add csrc/audio/speaker_verify/ecapa.h csrc/audio/speaker_verify/ecapa.cpp csrc/audio/speaker_gallery.h csrc/audio/speaker_gallery.cpp tests/test_speaker_verify.cpp tests/CMakeLists.txt
git commit -m "feat(audio): SpeakerVerify ECAPA core + SpeakerGallery"
```

---

### Task 2: 测试注册（test_speaker_verify 进 TEST_SOURCES）

**Files:**
- Modify: `tests/CMakeLists.txt`（在 `TEST_SOURCES` 的 audio 相关处加 `test_speaker_verify.cpp`）

**Interfaces:**
- Consumes: `test_speaker_verify.cpp` (Task 1)
- Produces: `[speaker]` Catch2 标签在 `test_modeldeploy.exe` 可跑

- [ ] **Step 1: Edit `tests/CMakeLists.txt`**

找到 `TEST_SOURCES` 列表（含 `test_encryption.cpp / test_tracking.cpp / ...` 处），加一行 `test_speaker_verify.cpp`（仿现有 audio/其他 test 的列法）。

- [ ] **Step 2: Reconfigure + build**

Run: `cmake --build build_tdc_gpu --config Release --parallel 8`
Expected: 0 errors，`test_modeldeploy.exe` 重新链接

- [ ] **Step 3: Run**

Run: `cd build_tdc_gpu && .\bin\test_modeldeploy.exe "[speaker]"`
Expected: 2 pas / , Test A gallery pass

- [ ] **Step 4: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "test(audio): register test_speaker_verify"
```
（若与 Task 1 同批提交则跳过，见 Task 1 提交说明）

---

### Task 3: pybind SpeakerVerify + SpeakerGallery

**Files:**
- Create: `csrc/pybind/audio/speaker_verify_pybind.cpp`
- Modify: `csrc/pybind/main.cpp`（声明 + `#ifdef BUILD_AUDIO` 下注册）
- Modify: 根 `CMakeLists.txt`（`NOT BUILD_AUDIO` 时 `REMOVE_ITEM` 新增的 pybind 源——参照 kokoro 处理）

**Interfaces:**
- Consumes: `SpeakerVerify`/`SpeakerGallery` (Task 1)
- Produces: Python `modeldeploy.audio.SpeakerVerify` / `modeldeploy.audio.SpeakerGallery`

- [ ] **Step 1: Write `csrc/pybind/audio/speaker_verify_pybind.cpp`**

```cpp
#include "pybind/utils/utils.h"
#include "audio/speaker_verify/ecapa.h"
#include "audio/speaker_gallery.h"

namespace modeldeploy::audio {
    void bind_speaker_verify(pybind11::module& m) {
        pybind11::class_<speaker_verify::SpeakerVerify, BaseModel>(m, "SpeakerVerify")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](speaker_verify::SpeakerVerify& self, const std::vector<float>& samples) {
                     std::vector<float> emb;
                     self.predict(samples, &emb);
                     return emb;
                 }, pybind11::arg("samples"))
            .def("is_initialized", &speaker_verify::SpeakerVerify::is_initialized);

        pybind11::class_<SpeakerGallery>(m, "SpeakerGallery")
            .def(pybind11::init<>())
            .def("enroll", &SpeakerGallery::enroll, pybind11::arg("label"), pybind11::arg("embedding"))
            .def("remove", &SpeakerGallery::remove, pybind11::arg("label"))
            .def("match", &SpeakerGallery::match, pybind11::arg("embedding"), pybind11::arg("k") = 1)
            .def("size", &SpeakerGallery::size)
            .def("clear", &SpeakerGallery::clear);
    }
} // modeldeploy::audio
```

- [ ] **Step 2: Edit `csrc/pybind/main.cpp`** — 加声明 `void bind_speaker_verify(pybind11::module&);`（在 `namespace modeldeploy::audio` 块内），并在 `#ifdef BUILD_AUDIO` 下 `audio::bind_speaker_verify(audio_module);`（`bind_kokoro` 之后）。

- [ ] **Step 3: Edit 根 `CMakeLists.txt`** — 找到 `NOT BUILD_AUDIO` 分支移除 kokoro_pybind.cpp 的 `list(REMOVE_ITEM ...)`，加 `speaker_verify_pybind.cpp`。

- [ ] **Step 4: Build + Python smoke**

Run: `.bat` 构建 `build_py` → `python -c "import modeldeploy; from modeldeploy import audio; g=audio.SpeakerGallery(); g.enroll('a',[1.0,0.0,0.0]); print(g.match([0.99,0.0,0.0],1))"`
Expected: `[('a', 1.0)]`（或 score≈1）

- [ ] **Step 5: Commit**

```bash
git add csrc/pybind/audio/speaker_verify_pybind.cpp csrc/pybind/main.cpp CMakeLists.txt
git commit -m "feat(pybind): bind SpeakerVerify + SpeakerGallery"
```

---

### Task 4: CAPI SpeakerVerify（embed 入口）

**Files:**
- Modify: `capi/md_capi.h`（`MD_MODEL_KIND` 加 `MD_MODEL_SPEAKER_VERIFY`；声明 `md_audio_speaker_embed`）
- Modify: `capi/md_capi.cpp`（create/delete/clone 分发 + 新入口）
- Test: `tests/test_capi.cpp`（`[capi]` speaker 用例）

**Interfaces:**
- Consumes: `SpeakerVerify` (Task 1)
- Produces: C API `MD_MODEL_SPEAKER_VERIFY` + `md_audio_speaker_embed(h, samples, n, &emb, &emb_n)`（借用指针）

- [ ] **Step 1: `capi/md_capi.h`** — 枚举在 `MD_MODEL_REID` 后加：
```cpp
MD_MODEL_SPEAKER_VERIFY,
```
（在 `MD_MODEL_COUNT` 前；实现时按当期 count 值为准追加）
声明：
```cpp
MD_CAPI_EXPORT MDStatus md_audio_speaker_embed(MDModelHandle h, const float* samples, size_t n,
                                  const float** embedding, size_t* emb_n);
```

- [ ] **Step 2: `capi/md_capi.cpp`** — 三处 `#ifdef BUILD_AUDIO` 分发：
- create：`case MD_MODEL_SPEAKER_VERIFY: { auto* m = new audio::speaker_verify::SpeakerVerify(parts[0], opt); if (!m->is_initialized()) return fail_init("SpeakerVerify"); *out = new md_model_handle{model, m}; } break;`
- delete：`case MD_MODEL_SPEAKER_VERIFY: delete static_cast<audio::speaker_verify::SpeakerVerify*>(model); break;`
- clone：`case MD_MODEL_SPEAKER_VERIFY: cloned = static_cast<audio::speaker_verify::SpeakerVerify*>(src->model)->clone().release(); break;`

- [ ] **Step 3: 新入口 `md_audio_speaker_embed`**（在 md_audio_tts 之后，仿 asr 入口结构）：
```cpp
MDStatus md_audio_speaker_embed(MDModelHandle h, const float* samples, size_t n,
                                const float** embedding, size_t* emb_n) {
    if (!h || !h->model) return {MD_ERROR_INVALID, "null handle"};
    if (!samples || n == 0) return {MD_ERROR_INVALID, "null/empty samples"};
    if (!embedding || !emb_n) return {MD_ERROR_INVALID, "null out"};
#ifdef BUILD_AUDIO
    auto* m = static_cast<audio::speaker_verify::SpeakerVerify*>(h->model);
    auto emb = std::make_shared<std::vector<float>>();
    if (!m->predict(std::vector<float>(samples, samples + n), emb.get()))
        return {MD_ERROR_RUNTIME, "speaker embed predict failed"};
    set_raw_embedding(h, emb);   // 参照现有 reid/face embedding 借用指针管理机制
    *embedding = emb->data();
    *emb_n = emb->size();
    return MD_SUCCESS;
#else
    return {MD_ERROR_RUNTIME, "built without BUILD_AUDIO"};
#endif
}
```
> 借用指针生命周期：对照 `md_result_face_embedding`/`md_result_reid_embedding` 在 md_capi.cpp 中的实现（结果句柄私有字段 + `md_result_destroy` 释放）。`set_raw_embedding` 为占位命名，**务必按现有 reid embedding 的实际内存管理机制改写成一致**。

- [ ] **Step 4: `tests/test_capi.cpp`** — 新增 `[capi]` speaker 用例：模型缺失 guard 跳过；枚举创建 `MD_MODEL_SPEAKER_VERIFY` 校验错误路径。

- [ ] **Step 5: Build + test**

Run: `.bat` 构建 `build_tdc_gpu` → `.\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；speaker 用例（模型缺失时 SKIP，枚举创建错误路径通过）

- [ ] **Step 6: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): MD_MODEL_SPEAKER_VERIFY + md_audio_speaker_embed"
```

---

### Task 5: C# SpeakerVerifyModel

**Files:**
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（`MDModelKind` 加 `MD_MODEL_SPEAKER_VERIFY`；extern `md_audio_speaker_embed`）
- Modify: `csharp/ModelDeploy/Models.cs`（新增 `SpeakerVerifyModel`）
- Modify: `csharp/ModelDeploy/NativeMethods.cs`（extern 导入）
- Test: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`SpeakerVerify_Works`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: C# `ModelDeploy.SpeakerVerifyModel`

- [ ] **Step 1: `types_internal_c.cs`** — `MDModelKind` 枚举加 `MD_MODEL_SPEAKER_VERIFY`（值对齐 CAPI）；`MDResultKind`/相关借用语义注释无需改（embed 直接返回）。

- [ ] **Step 2: `NativeMethods.cs`/`Models.cs`** — 新增 `SpeakerVerifyModel`：
```csharp
public class SpeakerVerifyModel {
    private readonly IntPtr _handle;
    public SpeakerVerifyModel(string modelPath) { /* md_model_create(MD_MODEL_SPEAKER_VERIFY, ...) */ }
    public float[] Predict(float[] samples) {
        // md_audio_speaker_embed -> Marshal.Copy 立即复制（借用指针）
    }
    public void Dispose() { /* md_model_destroy */ }
}
```
（对照 `ReIdModel`/`Kokoro` 的确切实现与 dispose 模式。）

- [ ] **Step 3: `AllModelsTests.cs`** — `[Fact]` `SpeakerVerify_Works`：构造 + Predict（无权重时 Skips）。

- [ ] **Step 4: Build + test**

Run: `dotnet build`（工作树 `csharp/`）；`dotnet test --filter SpeakerVerify_Works`
Expected: 0 errors；无权重 SKIP

- [ ] **Step 5: Commit**

```bash
git add csharp/ModelDeploy/*.cs csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(csharp): SpeakerVerifyModel + bindings"
```

---

### Task 6: Rust SpeakerVerify

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（`MDModelKind` 加 `SpeakerVerify`；extern 声明 `md_audio_speaker_embed`）
- Modify: `rust/modeldeploy/src/model.rs`、`types.rs`、`lib.rs`（`SpeakerVerify` 封装）
- Test: `rust/modeldeploy/tests/integration_test.rs`（`test_speaker_verify`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: Rust `modeldeploy::SpeakerVerify`

- [ ] **Step 1: `ffi.rs`** — 枚举加 `SpeakerVerify`（值对齐 CAPI）；extern 声明 `md_audio_speaker_embed`（签名 `(h, samples, n, &mut *const c_float, &mut usize)`）。

- [ ] **Step 2: `model.rs`/`types.rs`/`lib.rs`** — `SpeakerVerify` 封装（仿 `ReID` 的 `model_wrapper!` 或 `ReIdModel`）：`new(model, opt)` → `md_model_create(MD_MODEL_SPEAKER_VERIFY, ...)`；`predict(samples: &[f32]) -> Result<Vec<f32>>` → 调 `md_audio_speaker_embed`，`read_f32` 复制（安全）。

- [ ] **Step 3: `integration_test.rs`** — `#[test] fn test_speaker_verify`：权重缺失 skip；有权重时 `predict` 返回非空向量。

- [ ] **Step 4: Build + test**

Run: `cargo build`（`MODELDEPLOY_LIB_DIR` 指向 `build_tdc_gpu/bin`）；`cargo test test_speaker_verify`
Expected: 0 errors / clippy 干净；无权重 skip

- [ ] **Step 5: Commit**

```bash
git add rust/modeldeploy/src/*.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(rust): SpeakerVerify binding + test"
```

---

### Task 7: demo + docs

**Files:**
- Create: `examples/demo_speaker/demo_speaker.cpp`
- Create: `examples/demo_speaker/CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`（`add_subdirectory(demo_speaker)`）
- Modify: `examples/EXAMPLES.md`（加行）
- Modify: `README.md`（能力加"声纹/说话人验证"）

**Interfaces:**
- Consumes: `SpeakerVerify`/`SpeakerGallery` (Task 1)

- [ ] **Step 1: `examples/demo_speaker/CMakeLists.txt`**
```cmake
add_executable(demo_speaker demo_speaker.cpp)
target_link_libraries(demo_speaker PRIVATE ${LIBRARY_NAME})
```

- [ ] **Step 2: `demo_speaker.cpp`** — 加载模型 + 两个 wav（用 `csrc/utils/wave_helper.h` 的 `load_wav_file` 读 float PCM），各提 embedding，enroll 一个、match 另一个 → 打印 `(label, score)`。无权重/坏 wav 时清晰错误不崩溃；缺参 Usage。

- [ ] **Step 3: `examples/CMakeLists.txt`** 加 `add_subdirectory(demo_speaker)`；`EXAMPLES.md` 加 3 列行；`README.md` 能力列表加"声纹/说话人验证"。

- [ ] **Step 4: Build + run**

Run: `.bat` 构建 `build_tdc_gpu` → `.\bin\demo_speaker.exe`（无参 Usage / 缺模型错误路径）
Expected: 编译 0 errors；Usage/错误路径正常

- [ ] **Step 5: Commit**

```bash
git add examples/demo_speaker/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(examples): demo_speaker + docs"
```

---

### Task 8: 全量验证 + 跨后端语义确认

**Files:** 无新增（验证 + 必要时 minor 修复）

**Interfaces:**
- Consumes: 全部前序任务

- [ ] **Step 1: 全量 C++ 测试** — `.\bin\test_modeldeploy.exe "[speaker]"`、`"[capi]"`、`"[core]"`、`"[tracking]"`、`"[barcode]"`。记录通过数；speaker Test A 恒绿，其余无回归（capi barcode QR 失败 = 预存 `qr_sample.png` 缺失）。

- [ ] **Step 2: 跨后端语义确认** — grep 确认 `ecapa.cpp`/`speaker_gallery.cpp` 无 backend 直接依赖（仅 BaseModel）→ "各后端(ORT/MNN/TRT/Sophgo)语义一致"。ECAPA 输入布局若与 `[1,80,T]` 不符，按 `get_input_info(0).shape` 校正 preprocess 转置并更新注释。

- [ ] **Step 3: 绑定量测** — Python(import+SpeakerGallery+SpeakerVerify 构造冒烟)、C#(`SpeakerVerify_Works`)、Rust(`test_speaker_verify`)、demo_speaker 路径。

- [ ] **Step 4: 收尾报送** — 报告 + concerns（如 enum 值、embedding 借用指针与 reid 的一致性确认）。仅在确有必要且低风险时提交 bugfix。

---

## Self-Review 记录

**Spec coverage**：spec 的 SpeakerVerify(§4.2)、SpeakerGallery(§4.3)、前端 mel(§4.4)、Python(§5)、CAPI(§6)、C#(§7)、Rust(§8)、demo(§9)、test(§10)、交付矩阵(§11) 全部有对应 Task（T1 核心、T2 注册、T3 pybind、T4 CAPI、T5 C#、T6 Rust、T7 demo、T8 verify）。YAGNI 决策（SpeakerGallery 的 CAPI/C# 面不做）在 T4/T5 中体现为"仅模型 embed"。

**Placeholder**：Task 4 的 `set_raw_embedding` 为**命名占位**，已显式要求"务必按现有 reid embedding 实际内存管理机制改写"。这是实现时需以现有代码为准的关键点，非未定义接口——保留提醒。

**Type consistency**：`predict(data, &embedding)` 签名在 T1/T3/T5/T6 一致；`match(embedding, k)` 一致；枚举名 `MD_MODEL_SPEAKER_VERIFY` 全链一致。

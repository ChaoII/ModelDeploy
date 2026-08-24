# 行人 Re-ID（OSNet，512-d Embedding）设计

**日期**：2026-08-22
**状态**：设计（待审）

## 1. 目标与背景

为 ModelDeploy 增加**行人重识别（Person Re-ID）**能力：以 OSNet 模型（标准 CNN 行人 re-id 模型）从裁剪行人图提取 **512 维 L2 归一化**特征向量，用于跨摄像头行人匹配。可与已合并的 MOT 追踪家族（ByteTracker/BoT-SORT/StrongSORT，`TrackResult.feature` 已支持余弦关联）搭配。

**权重政策**：代码进仓库，权重外链（modelscope `test_data`）。

**现状**：`csrc/vision/tracking/reid_extractor.{h,cpp}` 已有 C++ 内部 L2 归一化特征提取器（仅被 BoT-SORT/StrongSORT 内部使用，未暴露任何语言面）；追踪器 `Detection.feature` / `TrackResult.feature` 已存在并被余弦关联消费。本设计在其上提供**独立、公开**的 Re-ID 模型 + 内存 Gallery 匹配。

## 2. 范围决策（已确认）

- **独立 `ReID` 模型类**（正式推荐 Option B），镜像 `SeetaFaceID`（face_rec）的 embedding 管线，不复用分类包装。
- **内存式 `ReIdGallery`**：注册 `label → embedding`，`match(embedding, k)` 返回 top-k 余弦最近邻。

## 3. 架构与组件

```
csrc/vision/reid/
├── reid.h/.cpp            ReID 模型类（predict/batch_predict/clone）
├── preprocessor.h/.cpp    专用预处理器（256x128，BGR->RGB CHW 归一化）
├── postprocessor.h/.cpp   后处理器（L2 归一化 512-d 输出）
└── gallery.h/.cpp         内存式 ReIdGallery（注册 + top-k 余弦匹配）
```

### ReID 模型类（`csrc/vision/reid/reid.h`）

```cpp
namespace modeldeploy::vision::reid {
struct ReIdResult { std::vector<float> embedding; };   // 512-d, already L2-normalized

class MODELDEPLOY_CXX_EXPORT ReID : public BaseModel {
public:
    ReID(const std::string& model_file, const RuntimeOption& option = RuntimeOption());

    bool predict(const ImageData& img, std::vector<ReIdResult>* results, TimerArray* timer = nullptr);
    bool batch_predict(const std::vector<ImageData>& imgs, std::vector<std::vector<ReIdResult>>* results, TimerArray* timer = nullptr);

    ReIDPreprocessor* get_preprocessor();
    ReIDPostprocessor* get_postprocessor();
    std::unique_ptr<ReID> clone() const;
};
}
```

- 预处理器：固定 `256x128`，`BGR→RGB CHW`，ImageNet 归一化（alpha/beta），镜像 `reid_extractor.cpp:50-76` 的 OSNet 布局。
- 后处理器：reshape 输出 → 512-d `embedding`，`utils::l2_normalize`（`utils.cpp:471`）。
- `initialize()` 做处理器后端选择（CPU/GPU/Sophgo），镜像 `Classification`。

### ReIdGallery（`csrc/vision/reid/gallery.h`）

```cpp
class ReIdGallery {
public:
    void clear();
    void enroll(const std::string& label, const std::vector<float>& embedding);  // adds/clobbers label
    std::vector<bool> remove(const std::string& label);
    std::vector<std::pair<std::string,float>> match(const std::vector<float>& embedding, int k) const; // top-k, 1-cosine
    size_t size() const;
};
```

- 内存 `map<string, vector<float>> gallery_`（embedding 已 L2 归一化）。
- `match` 用 `utils::compute_similarity`（点积，适合 L2 归一化向量），`cosine distance = 1 - dot`，降序取 top-k。
- 纯内存、线程安全（不加锁说明——调用方决定并发策略，镜像现有 demo 用法）。

## 4. 数据流

1. 输入裁剪 `ImageData`（行人框 crop）→ `ReID::predict` / `batch_predict`
2. preprocessor（256x128 CHW 归一化）
3. `infer`（BaseBackend → ORT/MNN/TRT/SOPHGO）
4. postprocessor（L2 归一化）→ 512-d `ReIdResult.embedding`
5. 可选：`ReIdGallery.enroll(...)` → `ReIdGallery.match(embedding, k)` → top-k `(label, score)`

与追踪器配对：`ReID::predict` 输出可赋给 `Detection.feature` / `TrackResult.feature`（已由 BoT-SORT/StrongSORT 余弦消费，维度灵活）。

## 5. 六面集成（对照 SeetaFaceID/Classification 模板）

| 面 | 注册点 |
|---|---|
| C++ core | `csrc/vision/reid/*.cpp` 自动 GLOB；`csrc/vision.h` 加 `#include "vision/reid/reid.h"` |
| pybind | `csrc/pybind/vision/reid_pybind.cpp`（`bind_reid`）；`vision_pybind.cpp` 声明+调用；`ReIdResult`/`ReIdGallery` 绑定 |
| CAPI | `capi/md_capi.h` 加 `MD_MODEL_REID`（`MD_MODEL_COUNT` 前）；create/destroy/clone/predict 分支；`md_result_reid_embedding`（镜像 `md_result_face_embedding`，`md_capi.cpp:2421`） |
| C# | `Models.cs` 加 `ReIdModel`（镜像 `FaceRecModel`）；`enum_varaibles.cs` 加 `MD_MODEL_REID`；`NativeMethods.cs` 加 getter |
| Rust | `model.rs` 加 `ResultType for ReID` + `model_wrapper!` + `reid()`；`types.rs` 加 `ModelKind::ReId` + `ReIdResult`；`ffi.rs` 加 `MD_MODEL_REID` |
| demo+docs+tests | demo（enroll + match 流程）；`EXAMPLES.md`；`tests/test_reid.cpp`（`[reid]`）；README 能力含 Re-ID |

## 6. API 一致性

- 结果类型：`ReIdResult { embedding: vector<float> }`（512-d L2 归一化）。
- 模型名/枚举：C++ `ReID`、CAPI `MD_MODEL_REID`、C# `ReIdModel`、Rust `ReID` / `ModelKind::ReId`、Python `ReID`。
- 方法：`predict` / `batch_predict` / `clone` 与现有模型对齐；`ReIdGallery` 提供 `enroll`/`match`/`remove`/`clear`/`size`。

## 7. 测试策略

- `tests/test_reid.cpp`（`[reid]`）：加载 OSNet ONNX，验证 embedding 维度 512 且近似 L2 归一化（\|v\|≈1）；Gallery enroll 两样本 + match 返回正确 top-1。数据 from modelscope `test_data`（外链）。
- CAPI `[capi]`：`MD_MODEL_REID` create/predict/embedding getter 契约。
- pybind/C#/Rust：import/调用冒烟 + embedding 断言 + Gallery 逻辑。
- 复用现有 `reid_extractor` 的 L2 归一化用例作为参考。
- 全量回归：`[core]`/`[tracking]`/`[barcode]` 无回归。

## 8. 明确不做（YAGNI）

- **不做** CAPI/C#/Rust 追踪器 per-detection feature 注入（当前 CAPI tracker 不接受 appearance feature），留作后续。
- 不做持久化 Gallery（纯内存）。
- 不做训练/模型微调。
- 不新建分类包装（独立 ReID 模型）。

## 9. 风险与备注

- CAPI `MDModelKind` 是稳定 ABI 枚举 —— 新值追加在 `MD_MODEL_COUNT` 之前，向后兼容。
- Sophgo int8 bmodel 通常 batch=1 静态形状 —— 与既有模型相同处理。
- 需要 OSNet ONNX 权重 —— 外链 modelscope `test_data`，测试数据单独下载。
- `utils::compute_similarity` 是纯点积（需输入已 L2 归一化）—— 后处理器已保证，Gallery 入参 enroll 前也应由上层归一化（文档注明）。

# 视频动作识别（TSN / ST-GCN 骨骼动作）——设计规范

- 日期：2026-08-22
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 6（顺序 4→5→7→6→8；**依赖 Item 7**：视频硬解下沉 + DAG 编排）
- 核心目标：在 Item 7 提供的 `VideoDecoder`（抽帧）与 `Dag`（编排）之上，为 SDK 新增**视频动作识别**能力——支持两类典型模型：TSN（RGB 帧时序聚合）与 ST-GCN（骨骼关键点时序图卷积）。SDK `csrc/` 目前无多帧序列/图模型支持，本 Item 补齐。

---

## 1. 背景与动机

探索确认：SDK `csrc/` 无视频/多帧序列输入（`tsn|st-gcn|skeleton|video|sequence` 无命中），`ImageData`/`BaseModel` 仅支持**单帧**。Item 7 将提供：
- `VideoDecoder`（`csrc/video/`，FFmpeg 软解 + 可插拔硬解）——抽帧。
- `Dag` / `Planner`（`csrc/pipeline/`）——编排。
- 姿态模型 `UltralyticsPose`（`csrc/vision/detection/ultralytics_pose.h`，关键点已成熟）——ST-GCN 的上游骨架来源。

本 Item 在此之上新建**动作识别模型类**，支持 TSN（RGB）与 ST-GCN（骨骼）两类主流模式。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope
- **TSN 动作识别**：多帧 RGB → 时序聚合 → 动作类别。采用"平均池化融合分段帧"的轻量实现（依赖 Item 7 视频抽帧给出帧序列）。
- **ST-GCN 骨骼动作识别**：姿态关键点序列 → 图卷积 → 动作类别。复用 `UltralyticsPose` 做骨架上游。
- 时序/图模型输入支持：`Tensor` shape 支持 `[N,C,T,H,W]`（TSN）与 `[N,C,T,V]`（ST-GCN）。验证现有 Tensor 是否支持 5D/4D 图输入，不足则扩展。
- 6 面贯通按能力降级（见 §10）。

### Out-of-Scope（明确不做，YAGNI）
- 不做在线实时动作识别（先做**离线/分段**：给定片段 → 类别）。
- 不做多段时序建模（LSTM/transformer 动作）、不做 3D-CNN 大模型（I3D/SlowFast，权重/算力重）。
- 不做动作检测（时间定位，action localization）；仅动作**分类**。
- ST-GCN 仅吃姿态关键点（不做多人的复杂图关联；单/固定人裁剪输入）。

## 3. 架构与组件

### 3.1 目录
```
csrc/vision/action/
    tsn.h / .cpp              # TSN : BaseModel
    st_gcn.h / .cpp           # StGcn : BaseModel
    action_ocr.h / .cpp       #（可选）动作类别标签映射
```
命名空间 `modeldeploy::vision::action`。

### 3.2 Tensor 图/时序输入支持（前置）
- 检查 `csrc/core/tensor.h` 的 `Tensor` 是否支持任意维（`std::vector<int64_t> shape_` 或 `ndims`）。若为固定 max 维（如 4D），扩展为支持 5D（TSN）与 4D 作为图骨架（ST-GCN 实为 `[N,C,T,V]`，4D 即可）。
- **本计划以 4D 优先**：TSN 把帧序列在 `preprocess` 内逐帧提取特征再 concat 为输入，或要求模型输入 `[N,C*T,H,W]` 的 uni-dimention（若模型如此）→ 以实际 ONNX 输入 shape 为准动态适配。ST-GCN 用 `[N,C,T,V]` 4D。

### 3.3 `TSN : BaseModel`
```cpp
namespace modeldeploy::vision::action {
class MODELDEPLOY_CXX_EXPORT TSN : public BaseModel {
public:
    TSN(const std::string& model_file,
        const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "TSN"; }
    // 输入：抽帧后的帧序列（RGB）；输出：类别 scores
    bool predict(const std::vector<ImageData>& frames, std::vector<float>* scores);
    [[nodiscard]] std::unique_ptr<TSN> clone() const;
    [[nodiscard]] bool is_initialized() const;
protected:
    bool initialize();
    bool preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);
private:
    explicit TSN() = default;
};
} // namespace modeldeploy::vision::action
```
- `predict(vector<ImageData> frames)` 输入由调用方用 Item 7 的 `VideoDecoder` 抽帧得到（或从任意历史帧列表）。preprocess 对每帧做缩放/归一化后组装为模型输入张量（聚合方式以模型为准，默认沿时间维堆叠）。

### 3.4 `StGcn : BaseModel`
```cpp
namespace modeldeploy::vision::action {
class MODELDEPLOY_CXX_EXPORT StGcn : public BaseModel {
public:
    StGcn(const std::string& model_file,
          const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "StGcn"; }
    // 输入：骨骼序列 [T, V, 2/3]（关节 x,y[,z]）；输出：类别 scores
    bool predict(const std::vector<KeyPointSeq>& skeleton_seq, std::vector<float>* scores);
    [[nodiscard]] std::unique_ptr<StGcn> clone() const;
    [[nodiscard]] bool is_initialized() const;
protected:
    bool initialize();
    bool preprocess(const std::vector<KeyPointSeq>& seq, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);
private:
    explicit StGcn() = default;
};
}
```
- `KeyPointSeq`：时间序列上的关节坐标数组（轻量结构，`std::vector<FrameKps>`，`FrameKps` 含 `std::vector<Point2f> joints`）。上游由 `UltralyticsPose` 对每帧提取关键点再组序而来。
- preprocess 组装为 `[N,C,T,V]`（N=1，C=2/3，T=时间帧数，V=关节数）。

### 3.5 编排（结合 Item 7）
- 示例 DAG（ST-GCN）：`VideoDecoder(next) -> PoseNode(UltralyticsPose 逐帧) -> KeyPointSeqNode(组序) -> StGcnNode -> ActionLabelNode`。
- 示例 DAG（TSN）：`VideoDecoder(next, K 帧) -> TSNNode -> ActionLabelNode`。
- 用 `Dag::connect`/`Planner` 手工/DSL 组装；示例 demo 演示完整链路。

## 4. Python（pybind）
- `csrc/pybind/vision/action_pybind.cpp`：`bind_tsn` / `bind_st_gcn`，在 `vision_pybind.cpp` 注册（`bind_action` 或单独声明）。
- `bind_vision`（`main.cpp` 的 `#ifdef BUILD_VISION`）调用。
- `TSN(list_of_ImageData) -> List[float] scores`；`StGcn(list_of_skeleton) -> List[float]`。

## 5. CAPI
- `MD_MODEL_KIND` 新增 `MD_MODEL_TSN` / `MD_MODEL_ST_GCN`。
- `md_model_create` 分发；新入口：
  - `md_model_predict_sequence(h, imgs/seq, n, out)`：对帧序列/骨骼序列推理。
  - 复用现有 `MD_RESULT_*` 的 scores 返回机制（或新增 `md_result_action` 返回 (label, score) 列表）。
- 具体签名以实现时按现有 CAPI 惯例（`md_result_classification` 形态）对齐。

## 6. C#
- `TsNModel` / `StGcnModel` 薄封装：`float[] Predict(IEnumerable<ImageData> frames)` / `Predict(float[] skeleton)`。枚举对齐 CAPI。

## 7. Rust
- `TsN` / `StGcn`（`model_wrapper!`）：`ffi.rs MDModelKind` 新增、extern 声明 sequence predict；`model.rs` 封装。

## 8. demo + docs
- `examples/demo_action/`：`demo_action.cpp + CMakeLists.txt`。加载行动模型 + 视频 → 打印动作类别（`BUILD_VIDEO` 用 VideoDecoder 抽帧；否则接受帧列表参数）。
- `EXAMPLES.md` 加行；README 能力加"视频动作识别"。

## 9. 测试
- `tests/test_action.cpp`（`[action]`）：TSN/StGcn 构造 + 合成输入 predict 的维度/类别断言（无权重 SKIP）。
- 桩输入测试：`TSN`、`StGcn` 的 preprocess 组装逻辑可用合成数据单测（不依赖权重）。
- Rust `test_tsn` / `test_st_gcn`、C# `Action_Works`（缺失权重 SKIP）。

## 10. 交付矩阵（6 面）

| 面 | TSN | ST-GCN |
|----|-----|--------|
| C++ | ✅ | ✅ |
| Python | ✅ | ✅ |
| CAPI | ✅ | ✅ |
| C# | ✅ | ✅ |
| Rust | ✅ | ✅ |
| demo+docs+tests | ✅ | ✅ |

> 取决于 Item 7 的交付：若 Item 7 未完成，TSN 的帧采集与 ST-GCN 编排退化为"调用方传帧/骨架数组"，模型类本身独立可测。

## 11. 已知限制 / 假设
- 权重外链；测试对缺失 SKIP。
- TSN 采用轻量平均池化聚合，不做复杂时序建模（YAGNI）。
- ST-GCN 固定单人/固定裁剪输入（多人图关联不在范围）。
- 依赖 Item 7 的 `VideoDecoder` 与 `Dag`；若未完成，模型类仍可独立交付并单测。

## 12. 风险
- Tensor 维数不足扩展（5D）：需确认 `core/tensor.h` 上限；优先 4D（STM 用 `[N,C,T,V]`），TSN 若模型要 5D 再扩展。
- TSN 模型输入布局多样；以实际 ONNX 输入 shape 动态适配。

## 13. 成功标准
- 模型类编译 + 桩输入测试可通过；无权重 SKIP。
- demo 在 `BUILD_VIDEO` 下演示视频→动作全链路。

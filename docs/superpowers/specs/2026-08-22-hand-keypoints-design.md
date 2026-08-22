# 手部关键点识别（Hand Keypoints，21 点）设计

**日期**：2026-08-22
**状态**：设计（待审）

## 1. 目标与背景

为 ModelDeploy 增加**手部关键点识别**功能：检测图像中的手部并输出 21 个关键点（MediaPipe 风格），可用于手势识别、AR、人机交互等。复用现有 Ultralytics 姿态（pose）管线——它已天然支持任意 N 关键点。

**模型权重政策**：代码进仓库，权重外链（modelscope `test_data`），不在仓库内。

## 2. 范围决策（已确认）

采用 **薄封装 `HandKeypoint` 包装 `UltralyticsPose`**：
- 新 `csrc/vision/hand/` 模块，内部持有一个 `UltralyticsPose`（`detection` 命名空间），默认 `set_keypoints_num(21)`。
- 更换为手部专属的 MediaPipe 21 骨架可视化 `vis_hand`。
- 不改动极简：复用 pose 通用管线、预/后处理、后端（ORT/MNN/TRT/SOPHGO）。

## 3. 架构与组件

```
csrc/vision/hand/
├── hand.h          HandKeypoint 模型类（薄包装 UltralyticsPose）
└── hand.cpp        predict/batch_predict/draw_result 委托给内部 pose_ 实例

csrc/vision/common/visualize/
└── vis_hand.cpp    新增 MediaPipe 21 骨架可视化
```

### HandKeypoint 类（`csrc/vision/hand/hand.h`）

```cpp
namespace modeldeploy::vision::hand {
class MODELDEPLOY_CXX_EXPORT HandKeypoint {
public:
    explicit HandKeypoint(const std::string& model_file, const RuntimeOption& option = RuntimeOption());

    // predict/batch_predict -> std::vector<KeyPointsResult>
    bool predict(const ImageData& img, std::vector<KeyPointsResult>* results, TimerArray* timer = nullptr);
    bool batch_predict(const std::vector<ImageData>& imgs, std::vector<std::vector<KeyPointsResult>>* results, TimerArray* timer = nullptr);

    void draw_result(ImageData& img, const std::vector<KeyPointsResult>& result, double threshold = 0.5);

    // 兼容 keypoints_num 配置（默认 21），暴露内层 pose 的预/后处理器
    vision::detection::UltralyticsPosePreprocessor* get_preprocessor();
    const ...* get_preprocessor() const;
    vision::detection::UltralyticsPosePostprocessor* get_postprocessor();
    const ...* get_postprocessor() const;

    std::unique_ptr<HandKeypoint> clone() const;
private:
    vision::detection::UltralyticsPose pose_;
};
}
```

实现要点：构造函数用 `model_file + option` 构造 `pose_`，随即 `pose_.get_postprocessor()->set_keypoints_num(21)`。`predict`/`batch_predict`/`draw_result` 直接委托内部 `pose_`。

### 可视化（`vis_hand.cpp`）

新增 `void vis_hand(const cv::Mat& im, const std::vector<KeyPointsResult>& result, double threshold, ...)`（镜像 `vis_pose`）。包含 MediaPipe 21 关键点连接表与调色板。关键点连接（CoCo → MediaPipe 序号见下）：
- 1-2,2-3,3-4（拇指 index 1-4）
- 0-5,5-6,6-7,7-8（食指）
- 5-9,9-10,10-11,11-12（中指）
- 9-13,13-14,14-15,15-16（无名指）
- 13-17,17-18,18-19,19-20（小指）
- 0-17（掌心根部）

## 4. 数据流

1. 输入 `ImageData` → `HandKeypoint::predict`
2. 内部 `UltralyticsPose` 的 preprocessor（letterbox+归一化，CPU/GPU/Sophgo 后端）
3. `infer`（Runtime/BaseBackend → ORT/MNN/TRT/SOPHGO）
4. postprocessor（N-generic 解码，`set_keypoints_num(21)`，去 letterbox padding → 输入像素坐标）
5. 输出 `vector<KeyPointsResult>`（`box` + `vector<Point3f> keypoints`，`Point3f.z` 存置信度）

属性与骨架在 `KeyPointsResult` 层面天然 21 点。

## 5. 六面集成（对照 UltralyticsPose 模板）

| 面 | 注册点 |
|---|---|
| C++ core | `csrc/vision/hand/*.cpp` 自动 GLOB（根 CMakeLists.txt）；`csrc/vision.h` 加入 `#include "vision/hand/hand.h"` |
| pybind | `csrc/pybind/vision/hand_pybind.cpp`（`bind_hand`）；在 `vision_pybind.cpp` 声明+调用；**补 `set_keypoints_num` 的 Python 暴露**（现 pose pybind 未绑） |
| CAPI | `capi/md_capi.h`：加 `MD_MODEL_HAND` 枚举（`MD_MODEL_COUNT` 前）；create/destroy/clone/predict 分支复用 pose 逻辑（`md_model_create`、`params`、`md_result_pose`/`md_result_keypoints`） |
| C# | `csharp/ModelDeploy/Models.cs` 加 `HandModel`（镜像 `PoseModel`，`SetKeypointsNum(v)`）；`enum_varaibles.cs` 加 `MD_MODEL_HAND`；`NativeMethods.cs` |
| Rust | `rust/modeldeploy/src/model.rs` 加 `ResultType for HandKeypoint`、`model_wrapper!`、`hand()`/`hand_batch()`；`types.rs` 加 `ModelKind::Hand`；`ffi.rs` 加 `MD_MODEL_HAND` |
| demo+docs+tests | `examples/demo_kps/` 加 `demo_hand`（或手部 demo 目录）；`EXAMPLES.md`；`tests/test_hand.cpp`（`[hand]`）；README 能力含手部 |

## 6. API 一致性

- 结果类型复用 `KeyPointsResult`（通用关键点结果，21 点即用）。
- 模型名/枚举：C++ `HandKeypoint`、CAPI `MD_MODEL_HAND`、C# `HandModel`、Rust `HandKeypoint` / `ModelKind::Hand`、Python `HandKeypoint`。
- 方法：`predict` / `batch_predict` / `draw_result` / `clone` 与现有模型对齐。

## 7. 测试策略

- `tests/test_hand.cpp`（`[hand]` 标签）：加载手部 ONNX，`set_keypoints_num(21)`，验证关键点数量 21、box 合理、置信度范围。数据 from modelscope `test_data/test_models/onnx/...`（外链）。
- CAPI `[capi]`：`MD_MODEL_HAND` create/predict 契约测试。
- pybind/C#/Rust：import/调用冒烟 + 单元断言。
- 全量回归：`[core]`/`[tracking]`/`[barcode]` 无回归。

## 8. 明确不做（YAGNI）

- 不做手势分类（仅关键点）。
- 不做 MediaPipe 之外的多手骨骼语义（骨架可视化仅绘制）。
- 不新建专用 pre/postprocessor —— 复用 pose 通用管线。
- 不处理训练。

## 9. 风险与备注

- `UltralyticsPose` 在 `detection` 命名空间 —— `HandKeypoint` 放 `hand` 命名空间内持有它，命名无冲突。
- Sophgo int8 bmodel 通常 batch=1 静态形状 —— 若用于 Sophgo 需与 pose 相同处理（如有 batch 需求 set_cls_batch_size 类调整）。
- 需要手部 ONNX 权重 —— 外链，测试数据单独下载。

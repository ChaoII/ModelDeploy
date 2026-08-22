# 关键点扩展（面部 Landmark / 车辆关键点）——设计规范

- 日期：2026-08-22
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 8（顺序 4→5→7→6→8）
- 核心目标：在既有 `UltralyticsPose`（姿态，17 点）、`HandKeypoint`（手部 21 点）之外，扩展关键点能力覆盖面——优先**面部 Landmark**（68/106 点，可由 InsightFace 的 2D106 覆盖）与**车辆关键点**。复用已建的 pose 骨架 `set_keypoints_num` 泛化机制。

---

## 1. 背景与动机

探索/既有实现确认视觉关键点现状：
- `UltralyticsPose`（`csrc/vision/detection/ultralytics_pose.*`）：人体 17 点，走 standard pose 后端；已泛化出 `set_keypoints_num()`（Item 2 在 hand 分支一并暴露到 pybind）。
- `HandKeypoint`（Item 2，`csrc/vision/hand/`）：手部 21 点，薄封装 `UltralyticsPose`。
- `SeetaFaceID` / InsightFace 的 `2d106det`（`csrc/vision/face/.../insightface_recognition.h`）：2D 106 点人脸关键点已存在（insightface 管线）。

因此本 Item 的落点是：
1. **面部 Landmark 独立模型**：把已有 InsightFace 106 点（或独立 68 点模型）暴露为**独立 `KeyPointModel` / 复用 `UltralyticsPose` 泛化**，供无需整 insightface 分析的用户直接使用。
2. **车辆关键点**：新增车辆关键点（如车轮/车窗 4-8 点）模型类，复用 pose 管线 `set_keypoints_num`。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope
- **面部 Landmark 独立访问**：将 insightface 2D106（或独立 68 点 ONNX）封装为可直接用的关键点模型；若沿用 `UltralyticsPose`/`HandKeypoint` 模式则复用 pose 后端。
- **车辆关键点**：新增 `VehicleKeypoint`（薄封装 `UltralyticsPose` + `set_keypoints_num`，如 4/8 点）。
- 复用 `KeyPointsResult`（`result.h`）、`vis_keypoints`/`vis_hand`（可视化）。
- 6 面贯通（见 §10）。

### Out-of-Scope（明确不做，YAGNI）
- 不做 3D 关键点（除 pose 已有 3D 字段外不新增 3D 模型）。
- 不重建 InsightFace 管线（已有 `insightface` 完整分析；本项只做"独立关键点"薄暴露）。
- 不做关键点后处理算法创新（一致性/可见性沿用现有）。

## 3. 架构与组件

### 3.1 目录
```
csrc/vision/landmark/
    face_landmark.h / .cpp       # FaceLandmark：独立人脸关键点
csrc/vision/landmark/
    vehicle_keypoint.h / .cpp    # VehicleKeypoint：车辆关键点
```
命名空间 `modeldeploy::vision::landmark`。

### 3.2 `FaceLandmark`（复用 pose 泛化）
```cpp
namespace modeldeploy::vision::landmark {
class MODELDEPLOY_CXX_EXPORT FaceLandmark : public BaseModel {
public:
    FaceLandmark(const std::string& model_file,
                 const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "FaceLandmark"; }
    // 输入：人脸裁剪图；输出：关键点结果（复用 KeyPointsResult）
    bool predict(const ImageData& image, std::vector<KeyPointsResult>* results);
    bool batch_predict(const std::vector<ImageData>& images, std::vector<KeyPointsResult>* results);
    [[nodiscard]] std::unique_ptr<FaceLandmark> clone() const;
    [[nodiscard]] bool is_initialized() const;
    void set_keypoints_num(int n);
    [[nodiscard]] int get_keypoints_num() const;
protected:
    bool initialize();
private:
    std::unique_ptr<detection::UltralyticsPose> pose_;
};
} // namespace modeldeploy::vision::landmark
```
- 策略：与 `HandKeypoint` 对称——薄封装 `UltralyticsPose`，`set_keypoints_num` 泛化到 68/106。若 face 模型结构与 pose 输出兼容则直接复用；否则内部改为专用实现（`initialize` 内按模型形状调整）。

### 3.3 `VehicleKeypoint`
```cpp
namespace modeldeploy::vision::landmark {
class MODELDEPLOY_CXX_EXPORT VehicleKeypoint : public BaseModel {
public:
    VehicleKeypoint(const std::string& model_file,
                    const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "VehicleKeypoint"; }
    bool predict(const ImageData& image, std::vector<KeyPointsResult>* results);
    bool batch_predict(const std::vector<ImageData>& images, std::vector<KeyPointsResult>* results);
    [[nodiscard]] std::unique_ptr<VehicleKeypoint> clone() const;
    [[nodiscard]] bool is_initialized() const;
    void set_keypoints_num(int n);
    [[nodiscard]] int get_keypoints_num() const;
protected:
    bool initialize();
private:
    std::unique_ptr<detection::UltralyticsPose> pose_;
};
} // namespace modeldeploy::vision::landmark
```
- 均输出 `KeyPointsResult`（复用 `result.h`），可视化可复用 `vis_keypoints`。

## 4. Python（pybind）
- `csrc/pybind/vision/landmark_pybind.cpp`：`bind_face_landmark` / `bind_vehicle_keypoint`，在 `vision_pybind.cpp` 注册（追加到 bind_hand/bind_reid 之后）。
- `set/get_keypoints_num` 暴露（与 pose 一样）。

## 5. CAPI
- `MD_MODEL_KIND` 新增 `MD_MODEL_FACE_LANDMARK` / `MD_MODEL_VEHICLE_KEYPOINT`。
- `md_model_create` 分发；结果复用 `MD_RES_POSE` 形态（`md_result_pose` / `md_result_keypoints`）——因输出同为 `KeyPointsResult`。

## 6. C#
- `FaceLandmarkModel` / `VehicleKeypointModel` 薄封装（仿 `HandModel`）：构造 + `Predict(ImageData)`。枚举对齐 CAPI。

## 7. Rust
- `FaceLandmark` / `VehicleKeypoint`（`model_wrapper!`）：`ffi.rs MDModelKind` 新增、`model.rs` 封装为 `ResultType=KeyPoints`（复用 pose keypoints 返回）。

## 8. demo + docs
- `examples/demo_landmark/`：`demo_landmark.cpp + CMakeLists.txt`——输入人脸/车辆图 → 打印关键点数量与坐标。
- `EXAMPLES.md` 加行；README 能力加"面部 Landmark / 车辆关键点"。

## 9. 测试
- `tests/test_landmark.cpp`（`[landmark]`）：`FaceLandmark`/`VehicleKeypoint` 构造 + `set/get_keypoints_num` 断言；真实推理缺权重 SKIP。
- Rust `test_face_landmark` / `test_vehicle_keypoint`、C# `Landmark_Works`（缺失权重 SKIP）。

## 10. 交付矩阵（6 面）

| 面 | FaceLandmark | VehicleKeypoint |
|----|--------------|-----------------|
| C++ | ✅ | ✅ |
| Python | ✅ | ✅ |
| CAPI | ✅ | ✅ |
| C# | ✅ | ✅ |
| Rust | ✅ | ✅ |
| demo+docs+tests | ✅ | ✅ |

## 11. 已知限制 / 假设
- 面部 Landmark 若复用 pose 后端，需模型输出形态与 `KeyPointsResult` 兼容（`set_keypoints_num` 泛化）；若 106 点模型结构不同，`initialize` 内做专用适配。
- 车辆关键点模型权重外链；测试对缺失 SKIP。
- 复用现有 `vis_keypoints`/`KeyPointsResult`，不新增结果结构。

## 12. 风险
- 面部 106 点模型是否可直接复用 pose 后端待实现时验证；若不可，回退为独立 `FaceLandmark` 专用实现（本 spec 已留该出口）。
- 车辆关键点点数/语义随模型而异，`set_keypoints_num` 应足够泛化。

## 13. 成功标准
- 模型类编译 + `set/get_keypoints_num` 逻辑测试通过；无权重 SKIP。
- demo 展示关键点输出；与手部/姿态能力形成完整"关键点"家族。

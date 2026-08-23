# 视觉应用解决方案层（CV Solutions）—— 设计规范

- 日期：2026-08-23
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 11（新增，紧随 Item 4/5/7/6/8 之后）
- 核心目标：借鉴 Ultralytics Solutions / supervision，在既有 C++ SDK（已有检测/分割/姿态/跟踪 ByteTrack/视频解码/关键点/动作能力）之上，编排出一层**面向真实应用场景的解决方案**，让 SDK 不再是空洞的底层推理壳，并充分发挥 C++ 性能。

---

## 1. 背景与动机

现状（已具备并可复用的地基）：
- 检测/分割/姿态/分类：`vision::detection/segmentation/pose/...`（yolo26n 全套 + 真实权重已在本机）。
- 多目标跟踪：`vision::tracking::ByteTracker`（`base_tracker.h`：`Detection{box,score,label_id,feature}` → `update(detections, frame, timestamp) → vector<TrackResult{track_id,box,...}>`；已有 pybind `tracking_pybind.cpp`）。
- 视频解码：`video::VideoDecoder`（FFmpeg 软解 → NV12 ImageData，`BUILD_VIDEO`）。
- 关键点：`vision::landmark`（Face/Vehicle）、`hand`、pose 关键点。

因此本 Item 落点：**在既有能力之上，以 C++ 编排出应用解决方案层**。这些方案主要是「组合 + 少量算法逻辑」，几乎不需要新权重（复用 yolo/姿态/跟踪权重点），纯 C++、性能强。

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
9. **VisionEye 可视化映射**（可选，若时间允许）：质心到"眼点"连线轨迹可视化。

### Out-of-Scope（明确不做，YAGNI）
- 不做 Streamlit 前端 / Analytics 图表渲染（偏前端，C++ 侧无价值）。
- 不做 Security Email 报警（依赖邮件服务，与应用无关）。
- 不做 Similarity Search-CLIP（需新 CLIP 权重 + 向量索引，超范围）。
- 不做 3D 测距 / 相机标定外参（除像素→米线性标定外不做相机模型）。
- 不做跟踪算法创新（复用现有 ByteTrack；SORT 为可选轻量补充）。

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

## 4. Python（pybind）
- `csrc/pybind/vision/solutions_pybind.cpp`：`vision.solutions` 子模块，绑定各方案类（构造/配置/update/draw/结果读取）。
- 注册进 `vision_pybind.cpp`（append 到既有之后），BUILD_VISION 门控。

## 5. CAPI
- 复用既有模型句柄体系还是新建 `solution` 句柄？**倾向新建轻量 `solution` C 风格句柄**（`md_solution_object_counter` 等），输入 track/detection 数组 → 输出结果。成本低、便于 C#/Rust 消费。
- 若工作量过大，CAPI/C#/Rust 可降级为 YAGNI（仅 C++/pybind/demo），在 plan 中明确。

## 6. C# / Rust
- 薄封装（若 CAPI 落地则顺带；否则降级 YAGNI）。

## 7. demo + docs
- `examples/demo_solutions/`：一个综合 demo，读视频/摄像头 → 检测 + ByteTrack → 计数/测速/热力图叠加 → 窗口显示；Workout/Parking 各做独立小 demo。
- README/EXAMPLES.md 能力行加「CV Solutions 应用方案（计数/热力图/测速/停车/健身等）」。

## 8. 测试
- `tests/test_solutions.cpp`（`[solutions]`）：每个方案用**合成轨迹/合成检测**做确定性断言（无权重）：
  - 计数：合成跨线轨迹 → 断言 in/out 计数。
  - 测速：合成等位移轨迹 + 时间戳 → 断言速度。
  - 热力图：合成轨迹 → 断言热度非零/峰值位置。
  - 裁剪/模糊：合成单框 → 断言尺寸/掩码。
  - 停车：合成框 vs 位元区域 → 断言占用状态。
  - 健身：合成肘角 → 断言动作计数。
- 真实权重路径（若有视频/模型）可选，无权重 SKIP。

## 9. 交付矩阵

| 面 | 覆盖 |
|----|------|
| C++ 核心 | ✅（9 方案） |
| Python | ✅（vision.solutions） |
| CAPI | ◐（可选，plan 裁量） |
| C#/Rust | ◐（若 CAPI 则随带；否则 YAGNI） |
| demo+docs+tests | ✅ |

## 10. 已知限制 / 假设
- 计数/测速精度依赖跟踪器稳定性（ByteTrack 已就位，作为限定）。
- 测速需合理时间戳（VideoDecoder 提供 pts_ms）+ 像素→米标定（用户提供 meter_per_pixel）。
- 健身计数基于肘/肩/腕角度阈值，为启发式（对齐 Ultralytics Workouts 简化版）。
- 停车需要用户定义位元区域。

## 11. 成功标准
- 各方案 C++ 类编译 + `[solutions]` 合成轨迹确定性测试通过（无权重也可验证）。
- demo_solutions 综合演示（计数/热力图/测速叠加）可跑（真实视频或摄像头）。
- SDK 从「底层推理」升级为「有应用场景方案」，补齐与 Ultralytics Solutions 的对应关系。

## 12. 对「SOTA 权重/真实测试」的安排
- 本 Item 聚焦应用层，不引入新 SOTA 权重；复用 yolo26n/姿态等已有真实权重点跑综合 demo。
- 其余待办（声纹 ecapa / ReID osnet / 手部 hand_pose / 动作 tsn/stgcn / 车辆 / 公式 / 文档真实权重下载与真测）单列一个「真实权重补全」后续项（可在本 Item demo 之后或并行进行，plan 里标注）。

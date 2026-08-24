# ModelDeploy 追踪家族（ByteTrack / BoT-SORT / StrongSORT）设计

日期：2026-08-21
状态：已批准
分支：`feature/tracking`

## 背景与目标

ModelDeploy 目前拥有完整的检测、分类、姿态、分割、OCR、人脸、LPR、音频等模块，但缺少**跨帧目标关联**能力。本设计为其补齐**多目标追踪（MOT）家族**，是 11 项功能扩展规划中的第一步。

目标：
- 纯 C++ 实现三种 SOTA 追踪器：**ByteTrack**（纯几何）、**BoT-SORT**（+Re-ID 外观 + 相机运动补偿）、**StrongSORT**（+Re-ID 外观 + NSA 卡尔曼 + EMA 特征库）
- 统一 Tracker 接口，输入每帧检测框列表，输出带稳定 `track_id` 的跟踪结果
- Re-ID 外观通过现有 ORT Runtime 加载 OSNet/ResNet ONNX 推理
- 全套语言绑定：C API / C# / Python / Rust
- 测试用 MOT 序列 + 指标断言（id 连续性、遮挡保持、HOTA/IDF1/MOTA）

## 架构

### 新目录 `csrc/vision/tracking/`

```
csrc/vision/tracking/
├── base_tracker.h/.cpp      // 统一接口 + 公共数据结构（Detection / TrackResult）
├── bytetrack.h/.cpp         // ByteTrack 实现
├── botsort.h/.cpp           // BoT-SORT 实现
├── strongsort.h/.cpp        // StrongSORT 实现
├── reid_extractor.h/.cpp    // ReID ONNX 封装（复用 Runtime）
├── matching/
│   ├── hungarian.h/.cpp     // 匈牙利指派求解器
│   ├── kalman_filter.h/.cpp // 卡尔曼滤波（linear 与 NSA 两版）
│   └── iou_matching.h/.cpp  // IoU / 外观代价矩阵构建
└── mot/
    ├── mot_loader.h/.cpp    // MOT 序列/检测结果加载
    └── clear_metrics.h/.cpp // CLEAR MOT 指标（HOTA/IDF1/MOTA）计算
```

vision 源码由根 CMake `file(GLOB_RECURSE VISION_SOURCE ...)` 自动收集，**无需修改 CMakeLists**；测试由 `BUILD_TESTS` 收集；绑定各加一个源文件。

### 核心数据结构

```cpp
struct Detection {            // 追踪输入（与 detection::DetectionResult 兼容）
    Rect2f box; float score; int label_id;
    std::vector<float> feature; // ReID 外观（可选，由 reid_extractor 填充）
};

struct TrackResult {          // 追踪输出
    int track_id;             // 跨帧稳定 ID
    Rect2f box; float score; int label_id;
    int state;                // New / Tracked / Lost / Removed
    std::vector<float> feature;
};
```

### 统一接口

```cpp
class BaseTracker {
public:
    // 输入检测框（纯几何追踪忽略 frame；外观追踪用它裁 patch 提取特征）
    virtual std::vector<TrackResult> update(
        const std::vector<Detection>& detections,
        const ImageData* frame = nullptr,
        double timestamp = -1) = 0;
    virtual void reset() = 0;
    virtual ~BaseTracker() = default;
    // 构造/配置：max_age, min_hits, iou_threshold, max_cost, track_thresh 等
};
```

三种实现共享 `BaseTracker` 接口，可互换。追踪器不是"模型+pre+post"三步，而是**后处理算法**——以接口而非模型类暴露。

## 数据流

```
UltralyticsDet.predict(image) -> std::vector<DetectionResult>
    -> 填充 Detection (box/score/label)
    -> tracker.update(detections, &frame)
        ├── ByteTrack  : 卡尔曼 + 匈牙利 + IoU 两阶段匹配（高分框 + 低分框）
        ├── BoT-SORT   : ByteTrack 基础 + ReID 余弦代价融合 + CMC(ECC) 位移补偿
        └── StrongSORT : NSA 卡尔曼 + EMA ReID 特征库 + 外观优先关联
    -> std::vector<TrackResult>  (含稳定 track_id)
```

## 三种追踪器算法（SOTA 对齐）

### ByteTrack（纯几何，无 ReID）
- 卡尔曼滤波（线性匀速假设）+ 匈牙利指派 + IoU 代价
- **核心创新**：低分检测框也参与匹配——每帧做"高置信框匹配" + "低置信框匹配"两阶段，抗遮挡/漏检
- 阈值：`track_thresh=0.5`（高低两档），`max_age=30`，`min_hits=3`

### BoT-SORT（ByteTrack 之上加外观）
- 第一阶段 IoU 匹配**融合外观余弦距离**（ReID 特征）构造代价矩阵
- **相机运动补偿（CMC）**：两帧间用 ECC 估计全局位移，匹配前校正预测框
- 依赖 ReID 提取器提供单目标特征

### StrongSORT（最重）
- **NSA Kalman**：噪声尺度自适应，抗状态突跳
- 外观匹配用 **EMA（指数移动平均）** 维护的 ReID 特征库 + 余弦距离
- 深度特征关联：外观优先于 IoU
- 依赖 ReID 提取器

## ReID 外观提取（reid_extractor）

- 封装 OSNet/ResNet 型 ONNX，输入单人裁剪图（由 `frame` + `Detection.box` 裁切）→ 输出特征向量
- 通过现有 `modeldeploy::Runtime` 加载，复用 `Tensor` / `ImageData` 预处理（resize + normalize）
- `std::vector<float> extract(const ImageData& patch)`
- 权重走 modelscope 下载至 test_data，不进仓库

## 语言绑定

- **Python**：`csrc/pybind/vision/tracking_pybind.cpp` — `ByteTracker`/`BotTracker`/`StrongTracker`（或统一 `create_tracker("bytetrack")`）+ `TrackResult`（含 track_id），`update(boxes)`
- **C API**：`capi/md_capi.h/.cpp` — `md_tracker_create/update/release`，track 结果数组
- **C#**：`csharp/ModelDeploy/Tracker.cs` + `NativeMethods.cs` DllImport + `TrackResult`
- **Rust**：`rust/modeldeploy/src/tracker.rs` 对应结构

## 测试

新增 `tests/test_tracking.cpp`：
- 加载 MOT 序列检测结果（或合成多帧轨迹）跑三种追踪器
- 断言：id 连续性、遮挡保持（max_age 内 track_id 复用）、外观相似/不同目标分辨
- 指标：轻量 CLEAR（HOTA/IDF1/MOTA）计算作为测试 helper
- 测试数据（MOT 序列 + ReID ONNX）走 modelscope/test_data，不进仓库

## 分支与提交策略

- 分支：`feature/tracking`（从 `release/v2.0.0` / `main` 分出）
- 一个算法一个提交，每步编译 + 测试通过后再提交：
  1. `feat(track): ByteTrack + BaseTracker 接口 + 匹配/卡尔曼组件`
  2. `feat(track): BoT-SORT（ReID 外观 + CMC）`
  3. `feat(track): StrongSORT（NSA Kalman + EMA）`
- 绑定与测试随各算法一起；全部完成后合并回 main

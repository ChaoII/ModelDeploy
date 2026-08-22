# 通用 Pipeline DAG 编排 + 视频硬解下沉 SDK ——设计规范

- 日期：2026-08-22
- 状态：已批准（brainstorming 一次规划）
- 路线：Item 7（顺序 4→5→7→6→8；本项是横切基建，先于 Item 6 视频动作）
- 核心目标：为 SDK `csrc/` 引入**通用 node/edge/planner DAG 编排层**，并把 `application/`（surveillance）里成熟的视频硬解/抽帧框架抽象为 **SDK 可复用组件**，为 Item 6（视频动作识别）及更复杂的多模型 pipeline 提供基础。

---

## 1. 背景与动机

探索确认：
- `csrc/` 内所有 "pipeline" 均是**硬编码的多模型顺序/并行组合**（`PaddleOCR` det→cls→rec，`PPStructureV2Table`，`pedestrian_attribute` det→classify，`lpr_pipeline`，`face_rec_pipeline`，`face_as_pipeline`）。**无通用 node/edge/planner DAG**。
- `application/`（`BUILD_SURVEILLANCE=ON` 的 surveillance 可执行程序）有**成熟的视频硬解/抽帧**（`stream_decoder.hpp` FFmpeg 软解 + CUVID 硬解 + Sophgo VPU）、`pipeline.hpp` 三段异步 decode→process→encode、`infer_group` 多模型 fan-out。但**隔离在可执行程序层，不在 SDK `csrc/`**。

本 Item 做两件事：① 通用 DAG 编排层；② 视频硬解抽帧下沉 SDK。为 Item 6（TSN/ST-GCN）与更复杂 pipeline 提供地基。

## 2. 范围（In-Scope / Out-of-Scope，YAGNI 收紧）

### In-Scope
- **通用 DAG 编排层**（单一职责的 node / edge / planner），支持：
  - 模型输出喂给另一模型输入（如 检测→跟踪→动作；版面→表格→SER）。
  - **单线程顺序/简单并行**执行；DAG 校验（无环、输入齐备）。
- **视频硬解/抽帧下沉 SDK**：把 `application/stream_decoder` 抽象为 `csrc/` 可复用组件（默认不带硬解 SDK 时退化为 FFmpeg 软解；CUVID/Sophgo 带设备实现保持可插拔）。
- 6 面贯通按能力对应（DAG 编排主要 C++/Python；视频硬解主要 C++；CAPI/C#/Rust 以能代表使用的薄封装为准——见 §10）。

### Out-of-Scope（明确不做，YAGNI）
- 不做多对多 / 异构后端 / 多进程并行 DAG（单线程 + 简单 fan-out 即可覆盖 Item 6 与当前管线）。
- 不做分布式调度、GPU 集群。
- 不做可视化 DAG 编辑器 / 动态热插拔。
- 不重写 `application/` 现有 surveillance（仅把可复用部分抽象下沉，surveillance 继续用它或保持独立）。

## 3. 架构与组件

### 3.1 目录
```
csrc/pipeline/
    node.h / .cpp        # Node：输入/输出端口、执行委托
    edge.h / .cpp        # Edge：连接 port（类型匹配校验）
    dag.h / .cpp         # Dag：节点/边注册、拓扑排序、执行
    planner.h / .cpp     # Planner：给定模型组合，自动布节点/边
csrc/video/
    frame_grabbing.h / .cpp  # 硬解/抽帧下沉组件（FFmpeg 软解默认；CUVID/Sophgo 可插拔）
```
- `csrc/*.cpp` GLOB_RECURSE 自动收集，无需改 CMake（探索确认行 110-124）。

### 3.2 `Node`（单一职责）
```cpp
namespace modeldeploy::pipeline {
struct Port {
    std::string name;
    std::string type;   // 逻辑类型名："Image" / "DetectionResult" / "TrackResult" / "Tensor" ...
};

class MODELDEPLOY_CXX_EXPORT Node {
public:
    Node(std::string name, std::vector<Port> in, std::vector<Port> out);
    virtual ~Node() = default;
    const std::string& name() const;
    const std::vector<Port>& inputs() const;
    const std::vector<Port>& outputs() const;
    // 子类实现：读取输入端口值，写入输出端口
    virtual bool run() = 0;
protected:
    template<typename T> bool get_in(const std::string& port, T* out);
    template<typename T> void set_out(const std::string& port, const T& v);
private:
    std::unordered_map<std::string, std::any> in_data_, out_data_;
};
} // namespace modeldeploy::pipeline
```
- `Node` 用 `std::any` 承载任意结果类型值，端口按名称取用。子类（模型节点、后处理节点）实现 `run()`。
- 提供便捷子类：`ModelNode<T>(T model, in/out ports)` 包装任意 BaseModel 的 predict；`CopyNode` / `TransformNode` 等通用节点可按需添加（YAGNI：先只做 ModelNode + 一个通用 TransformNode）。

### 3.3 `Dag`
```cpp
class MODELDEPLOY_CXX_EXPORT Dag {
public:
    void add_node(std::unique_ptr<Node> node);
    // src.out_port -> dst.in_port
    bool connect(const std::string& src_node, const std::string& src_port,
                 const std::string& dst_node, const std::string& dst_port);
    /// 拓扑排序：无环则返回执行序，否则 false。
    bool build();                                  // 校验：连通、类型匹配、无环
    bool execute();                                // 按拓扑序单线程执行
    std::vector<std::string> execution_order() const;
};
```
- 执行时按拓扑序调用各 node `run()`，把上游输出拷贝到下游输入端口。
- 校验：未连接输入报错；类型不匹配（端口 type 不同）报错；有环跳过该环上节点或报错（YAGNI：报错并返回 false，不做环消解）。

### 3.4 `Planner`（给定模型组合自动布线）
```cpp
class MODELDEPLOY_CXX_EXPORT Planner {
public:
    // 注册可用的模型节点工厂：名称 -> 构造回调（把模型实例包成 Node）
    void register_model(const std::string& name, const std::function<std::unique_ptr<Node>(const Variant&)>& factory);
    /// 用 DSL/描述构建 DAG：如 "det -> track -> action"
    std::unique_ptr<Dag> build(const std::string& spec);
};
```
- Planner 提供最小 DSL：`A -> B -> C` 表示顺序布线（端口名自动匹配唯一输入/输出）；`{A, B} -> C` 表示 fan-in（C 多输入），`A -> {B, C}` 表示 fan-out。
- **YAGNI：Planner 只支持顺序 + 单级 fan-in/fan-out，不做任意图构造。** 更复杂图可直接用 `Dag::add_node/connect` 手工搭。

### 3.5 视频硬解下沉 `FrameGrabbing`
```cpp
namespace modeldeploy::video {
struct Frame { /* 包装解码帧：NV12 或 BGR Mat，时间戳，设备信息 */ };
class MODELDEPLOY_CXX_EXPORT VideoDecoder {
public:
    VideoDecoder(const std::string& url_or_path, const DeviceOption& dev, bool hw_accel=false);
    bool open();
    /// 抽下一帧，并可选已解出为 ImageData（DeviceOption 指定设备）
    bool next(ImageData* out, uint64_t* pts_ms);
    void close();
    [[nodiscard]] int width() const; int height() const; double fps() const;
};
} // namespace modeldeploy::video
```
- 实现移植 `application/stream_decoder` 的核心，FFmpeg 软解为默认必选；CUVID（`hw_accel=="cuda"`）与 Sophgo VPU 以可选实现存在，未配置对应设备时优雅回退软解。
- **依赖**：引入 FFmpeg（已有 `application` 使用）；下沉后 SDK 链接 FFmpeg（可在 `BUILD_VIDEO=ON` 选项下启用，默认 OFF 避免提升编译门槛）。

## 4. Python（pybind）
- 新增 `csrc/pybind/pipeline/pipeline_pybind.cpp`：暴露 `pipeline.Dag` / `pipeline.Node`（ModelNode 便捷构造）/ `pipeline.Planner`。`build("det -> track -> action")` 后 `.execute()`。
- 新增 `csrc/pybind/video/video_pybind.cpp`：`video.VideoDecoder(url, ...)`，`.next()` 返回 `ImageData`（与 vision.ImageData 复用绑定）。仅在 `BUILD_VIDEO=ON` 下注册。
- 注册：`csrc/pybind/main.cpp` 增加 `pipeline` 子模块与 `video` 子模块（video 需 `#ifdef BUILD_VIDEO`）。

## 5. CAPI
- DAG/视频**以能代表使用的薄封装**为主（YAGNI）：`md_pipeline_create/build/execute`（字符串 DSL 建 DAG）+ `md_video_decoder_create/next/destroy`（`BUILD_VIDEO` 保护）。
- 底层仍复用同一 `Dag`/`VideoDecoder`。不暴露任意 Node 图构造到 CAPI（保持薄）。

## 6. C#
- 薄封装：`Pipeline`（`Build(string spec)` / `Execute()`）+ `VideoDecoder`（`Next()` → `ImageData`）。枚举/常量对齐 CAPI。DAG 内部结构不镜像。

## 7. Rust
- 薄封装：`pipeline::Dag::build(spec)?.execute()`；`video::VideoDecoder::open()?.next()?`。`ffi.rs` extern 声明 `md_pipeline_*` / `md_video_decoder_*`。

## 8. demo + docs
- `examples/demo_pipeline/` 扩展现有（或新增）demo：用 `det -> track -> ...` 演示 DAG 编排；`BUILD_VIDEO` 时演示 `VideoDecoder` 抽帧 → DAG 处理。
- `EXAMPLES.md` 加行；README 能力加"Pipeline DAG 编排 / 视频解码"。

## 9. 测试
- `tests/test_pipeline.cpp`（`[pipeline]`）：Dag 拓扑排序、环检测、端口连接/类型校验、execute 数据流（用桩 Node 不依赖权重）。
- `tests/test_video_decoder.cpp`（`[video]`，`BUILD_VIDEO`）：对合成/本地视频文件抽帧断言宽高/fps/帧数。无视频文件 SKIP。
- Rust `test_pipeline`、C# `Pipeline_Works`。

## 10. 交付矩阵（6 面）

| 面 | DAG 编排 | 视频硬解下沉 |
|----|---------|-------------|
| C++ | ✅（Node/Edge/Dag/Planner） | ✅（VideoDecoder） |
| Python | ✅ | ✅（BUILD_VIDEO） |
| CAPI | ✅（薄：build/execute） | ✅（薄：create/next） |
| C# | ✅（薄） | ✅（薄） |
| Rust | ✅（薄） | ✅（薄） |
| demo+docs+tests | ✅ | ✅ |

## 11. 已知限制 / 假设
- DAG 单线程顺序/简单并行；复杂并行/异构后端不在本计划。
- 视频硬解依赖 FFmpeg，经 `BUILD_VIDEO` 选项启用（默认 OFF）；未启用时 VideoDecoder 相关绑定/Python 子模块不编译。
- 端口类型为逻辑字符串（`"Image"`/`"DetectionResult"` 等），运行时 `std::any` 强转；类型不匹配在 connect 期据 type 名校验，运行时强转失败报错。

## 12. 风险
- `std::any` 多态结果传递的性能与类型安全：本计划用引用语义 + connect 期校验，满足单线程场景。
- 移植 stream_decoder 到 SDK 涉及 FFmpeg 链接与设备抽象；以"软解默认必选、硬解可插拔"控制风险。

## 13. 成功标准
- C++/Python/CAPI/C#/Rust 编译通过；桩 Node 的 DAG 测试恒定通过。
- 视频硬解测试在 `BUILD_VIDEO` + 有视频文件时通过，否则 SKIP。
- Item 6（视频动作识别）可基于本 DAG/抽帧组件搭建。

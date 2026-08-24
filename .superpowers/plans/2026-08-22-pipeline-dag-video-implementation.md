# Item 7: 通用 Pipeline DAG 编排 + 视频软解下沉 SDK ——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SDK 引入通用 node/edge/Dag/Planner DAG 编排层（`csrc/pipeline/`），并把现有 `application/stream_decoder` 的 **FFmpeg 软解核心** 下沉为 SDK 可复用 `video::VideoDecoder`（`csrc/video/`），C++/Python 为主要交付。

**Architecture:** 两条相对独立子系统。子系统 A（DAG）是纯 C++、无外部依赖：`Node` 用 `std::any` 承载端口数据，`Dag` 做拓扑排序/无环/类型校验/单线程 execute，`Planner` 解析最小 DSL（顺序 + 单级 fan-in/out）。子系统 B（视频）依赖 FFmpeg，经 `BUILD_VIDEO`（默认 OFF）启用，只做软解：`VideoDecoder::next` 复用 stream_decoder 的 FFmpeg 读帧/软解逻辑，把解码帧 NV12 双平面经 `ImageData::from_planes` 包装成 `vision::ImageData`。CAPI/C#/Rust 按 YAGNI 标注为可选/后续薄封装（不写全量）。

**Tech Stack:** C++17、FFmpeg（软解）、pybind11、Catch2、现有 `ImageData`/`Device`/`Tensor`。

**Spec:** `docs/superpowers/specs/2026-08-22-pipeline-dag-video-design.md`

## Global Constraints

- **子模块命名**：DAG 用 `csrc/pipeline/`（`node/edge/dag/planner.{h,cpp}`，namespace `modeldeploy::pipeline`）；视频用 `csrc/video/`（namespace `modeldeploy::video`）。
- **现状校正——视频类名**：目标态命名 `VideoDecoder`，方法 `open(url)/next(ImageData*, uint64_t* pts_ms)/close()/width()/height()/fps()`；底层复用 `application/stream_decoder.{hpp,cpp}` 的 **FFmpeg 软解核心**（h264/hevc 软解 + 双平面 NV12）。**不搬** surveillance 强耦合代码：Pipeline/TaskConfig/DrawEngine/StreamEncoder。
- **现状校正——YAGNI 硬解不搬**：`video::VideoDecoder` **只做 FFmpeg 软解**，不引入 CUVID/Sophgo/hw_accel（避免 SDK 引入 CUDA 依赖）。CUVID/Sophgo 留在 application/stream_decoder，不搬。
- **现状校正——无 DeviceOption**：用 `modeldeploy::Device` 枚举（`csrc/core/enum_variables.h:16`，CPU/GPU/OPENCL/VULKAN/TPU）。视频软解输出 CPU NV12，device 恒为 `Device::CPU`。
- **现状校正——CMake**：新增 `option(BUILD_VIDEO OFF)`；新建 `cmake/ffmpeg.cmake` 模块化 `application/CMakeLists.txt:11-22` 的 find 逻辑（`FFMPEG_ROOT` CACHE 变量，产出 `FFMPEG_INCLUDE_DIR` + `FFMPEG_LIBS`）；`BUILD_VIDEO` 时 `include(cmake/ffmpeg.cmake)` + 编译 `csrc/video/`（仿 `BUILD_AUDIO` 模式，根 `CMakeLists.txt:206-220`）。
- **现状校正——CMake GLOB 行为**：`file(GLOB_RECURSE ALL_SOURCE ... csrc/*.cpp)`（根 `CMakeLists.txt:110`）默认会收集 `csrc/pipeline/*.cpp` 与 `csrc/video/*.cpp`。`csrc/pipeline/` 保持永远编译（纯 C++，无外部依赖）。`csrc/video/` 依赖 FFmpeg，必须新增 `VIDEO_SOURCE` glob 并**加入第 124 行的 `REMOVE_ITEM ALL_SOURCE`**，再在 `BUILD_VIDEO` 下条件 append（否则 `BUILD_VIDEO=OFF` 时也会编译导致缺 FFmpeg 报错）。
- **现状校正——视频依赖 vision**：`VideoDecoder::next` 返回 `vision::ImageData`（`csrc/vision/common/image_data.h`），故 **`BUILD_VIDEO` 要求 `BUILD_VISION=ON`**（CMake 断言，缺则 FATAL_ERROR）；Python `video` 子模块同样受 `BUILD_VISION` 约束。
- **端口类型**：逻辑字符串（`"Image"`/`"DetectionResult"`/`"int"` 等），connect 期按 type 名校验，运行时 `std::any` 强转；强转失败 `run()` 返回 false。
- **Tensor/5D**：`Tensor.shape` 是 `std::vector<int64_t>`（无维数上限），DAG 数据为 `std::any`，不涉及 Tensor 维数问题。
- **无权重**：DAG 测试用桩 Node 恒定通过（不依赖权重）；视频测试需本地视频文件，找不到 `MD_TEST_VIDEO` 则 SKIP。
- **本机 FFmpeg**：`E:/develop/ffmpeg`；测试运行时 DLL（avcodec-*.dll 等）需在 PATH 或 FFmpeg bin 目录（否则视频测试链接通过但运行报 DLL not found —— 见 Task 6 步骤中的运行说明）。
- 复用 `csrc/core/md_decl.h` 的 `MODELDEPLOY_CXX_EXPORT`；C++17。
- **Python `video` 子模块复用 `vision.ImageData` 绑定**：ImageData 已在 `modeldeploy.vision` 子模块注册（`image_data_pybind.cpp`），同一 `modeldeploy` 扩展内类型注册表共享，`video_pybind` 返回 `ImageData` 时 pybind 自动复用 `vision.ImageData`。
- **C#/Rust（YAGNI 后续）**：本计划**不写** C#/Rust 全量代码；仅在「可选/后续」一节注明薄封装接口形态（`Pipeline.Build(spec)/Execute`、`VideoDecoder.Next()`；Rust `pipeline::Dag::build()?`/`video::VideoDecoder::open()?`）。**CAPI** 作为可选薄封装单列（Task 8，标注为可降级跳过）。

---

## 子系统 A：通用 DAG 编排层（纯 C++，无外部依赖）

### Task 1: `Node` + `Edge` 核心类 + 基础测试

**Files:**
- Create: `csrc/pipeline/node.h`
- Create: `csrc/pipeline/node.cpp`
- Create: `csrc/pipeline/edge.h`
- Create: `csrc/pipeline/edge.cpp`
- Test: `tests/test_pipeline.cpp`

**Interfaces:**
- Produces (later tasks rely on):
  - `struct modeldeploy::pipeline::Port { std::string name; std::string type; }`
  - `class Node`：`Node(std::string name, std::vector<Port> in, std::vector<Port> out)`；`virtual bool run() = 0`；`std::string name()`；`std::vector<Port> inputs()/outputs()`；`void set_input(const std::string& port, std::any v)`；`void clear_outputs()`；`bool has_input(const std::string& port) const`；`std::any get_output(const std::string& port) const`；`protected template<T> bool get_in(const std::string&, T*)`；`protected template<T> void set_out(const std::string&, const T&)`。
  - `struct modeldeploy::pipeline::Edge { std::string src_node, src_port, src_type, dst_node, dst_port, dst_type; }`
  - `bool edge_type_compatible(const Edge&)`（src_type == dst_type）

- [ ] **Step 1: Write failing tests** (`tests/test_pipeline.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/edge.h"
#include <any>

using namespace modeldeploy::pipeline;

// 桩：int 输入 -> int 输出（乘 2）。可派生子类。
struct DoubleNode : Node {
    DoubleNode(std::string n) : Node(std::move(n), {{"in", "int"}}, {{"out", "int"}}) {}
    bool run() override {
        int v;
        if (!get_in<int>("in", &v)) return false;
        set_out<int>("out", v * 2);
        return true;
    }
};

// 桩：Image 类型输入，无输出（用于类型不匹配连接的校验）。
struct ImageSinkNode : Node {
    ImageSinkNode(std::string n) : Node(std::move(n), {{"in", "Image"}}, {}) {}
    bool run() override { return true; }
};

// 桩：int 输入 -> int 输出（原样透传）。
struct IdentityNode : Node {
    IdentityNode(std::string n) : Node(std::move(n), {{"in", "int"}}, {{"out", "int"}}) {}
    bool run() override {
        int v;
        if (!get_in<int>("in", &v)) return false;
        set_out<int>("out", v);
        return true;
    }
};

TEST_CASE("Node port schema", "[pipeline]") {
    DoubleNode n("d");
    REQUIRE(n.name() == "d");
    REQUIRE(n.inputs().size() == 1);
    REQUIRE(n.inputs()[0].name == "in");
    REQUIRE(n.inputs()[0].type == "int");
    REQUIRE(n.outputs().size() == 1);
    REQUIRE(n.outputs()[0].type == "int");
}

TEST_CASE("Node set/get data via any", "[pipeline]") {
    DoubleNode n("d");
    REQUIRE_FALSE(n.has_input("in"));
    n.set_input("in", std::any(3));
    REQUIRE(n.has_input("in"));
    REQUIRE(n.run());
    REQUIRE(std::any_cast<int>(n.get_output("out")) == 6);
}

TEST_CASE("Edge type compatibility", "[pipeline]") {
    Edge same{"a", "out", "int", "b", "in", "int"};
    REQUIRE(edge_type_compatible(same));
    Edge diff{"a", "out", "int", "b", "in", "Image"};
    REQUIRE_FALSE(edge_type_compatible(diff));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: FAIL（`csrc/pipeline/node.h` 不存在，编译失败）

> 先执行 Task 1 Step 3/4 创建文件的同一次构建。若 `build` 目录尚未配置，先按 AGENTS.md 配置：`cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_TESTS=ON`。

- [ ] **Step 3: Write `csrc/pipeline/edge.h`**

```cpp
#pragma once

#include <string>

namespace modeldeploy::pipeline {

// 一条有向边：src 节点 out_port -> dst 节点 in_port，记录两端端口 type 供一致性校验。
struct Edge {
    std::string src_node;
    std::string src_port;
    std::string src_type;
    std::string dst_node;
    std::string dst_port;
    std::string dst_type;
};

// src_type 与 dst_type 一致才算类型兼容
bool edge_type_compatible(const Edge& edge);

} // namespace modeldeploy::pipeline
```

- [ ] **Step 4: Write `csrc/pipeline/edge.cpp`**

```cpp
#include "csrc/pipeline/edge.h"

namespace modeldeploy::pipeline {

bool edge_type_compatible(const Edge& edge) {
    return edge.src_type == edge.dst_type;
}

} // namespace modeldeploy::pipeline
```

- [ ] **Step 5: Write `csrc/pipeline/node.h`**

```cpp
#pragma once

#include "core/md_decl.h"
#include <any>
#include <string>
#include <unordered_map>
#include <vector>

namespace modeldeploy::pipeline {

// 端口：名称 + 逻辑类型名（"Image"/"DetectionResult"/"int" ...）
struct Port {
    std::string name;
    std::string type;
};

// 单一职责：声明输入/输出端口，run() 读取输入写入输出。数据用 std::any 承载。
class MODELDEPLOY_CXX_EXPORT Node {
public:
    Node(std::string name, std::vector<Port> in, std::vector<Port> out);
    virtual ~Node() = default;

    const std::string& name() const { return name_; }
    const std::vector<Port>& inputs() const { return in_; }
    const std::vector<Port>& outputs() const { return out_; }

    virtual bool run() = 0;

    // Dag 布线/种子使用：外部/上游写入输入；执行前置空输出；读取输出
    void set_input(const std::string& port, std::any v);
    void clear_outputs();
    bool has_input(const std::string& port) const;
    std::any get_output(const std::string& port) const;

protected:
    template <typename T>
    bool get_in(const std::string& port, T* out) {
        auto it = in_data_.find(port);
        if (it == in_data_.end() || !it->second.has_value()) return false;
        try {
            *out = std::any_cast<T>(it->second);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    template <typename T>
    void set_out(const std::string& port, const T& v) {
        out_data_[port] = std::any(v);
    }

private:
    std::string name_;
    std::vector<Port> in_, out_;
    std::unordered_map<std::string, std::any> in_data_, out_data_;
};

} // namespace modeldeploy::pipeline
```

- [ ] **Step 6: Write `csrc/pipeline/node.cpp`**

```cpp
#include "csrc/pipeline/node.h"

namespace modeldeploy::pipeline {

Node::Node(std::string name, std::vector<Port> in, std::vector<Port> out)
    : name_(std::move(name)), in_(std::move(in)), out_(std::move(out)) {}

void Node::set_input(const std::string& port, std::any v) {
    in_data_[port] = std::move(v);
}

void Node::clear_outputs() {
    out_data_.clear();
}

bool Node::has_input(const std::string& port) const {
    auto it = in_data_.find(port);
    return it != in_data_.end() && it->second.has_value();
}

std::any Node::get_output(const std::string& port) const {
    auto it = out_data_.find(port);
    if (it == out_data_.end()) return {};
    return it->second;  // 拷贝：std::any 要求类型可拷贝（int/ImageData 均满足）
}

} // namespace modeldeploy::pipeline
```

- [ ] **Step 7: Register test + build + verify pass**

Modify `tests/CMakeLists.txt` TEST_SOURCES: add `test_pipeline.cpp`.

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: 3 个 `[pipeline]` 用例全部 PASS。

- [ ] **Step 8: Commit**

```bash
git add csrc/pipeline/node.h csrc/pipeline/node.cpp csrc/pipeline/edge.h csrc/pipeline/edge.cpp tests/test_pipeline.cpp tests/CMakeLists.txt
git commit -m "feat(pipeline): Node + Edge core with port schema and type check"
```

---

### Task 2: `Dag`（拓扑排序 / 无环 / 输入齐备 / execute 数据流）

**Files:**
- Create: `csrc/pipeline/dag.h`
- Create: `csrc/pipeline/dag.cpp`
- Modify: `tests/test_pipeline.cpp`

**Interfaces:**
- Consumes: Node/Edge (Task 1)
- Produces (later tasks rely on):
  - `class Dag`：`void add_node(std::unique_ptr<Node>)`；`Node* get_node(const std::string&)`；`bool connect(src_node, src_port, dst_node, dst_port)`；`bool build()`；`bool execute()`；`std::vector<std::string> execution_order() const`；`const std::vector<Edge>& edges() const`。

- [ ] **Step 1: Append failing tests** (`tests/test_pipeline.cpp`)

```cpp
#include "csrc/pipeline/dag.h"
#include <deque>
#include <memory>

TEST_CASE("Dag execute dataflow sequential", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("d1"));
    dag.add_node(std::make_unique<DoubleNode>("d2"));
    REQUIRE(dag.connect("d1", "out", "d2", "in"));
    dag.get_node("d1")->set_input("in", std::any(3));
    REQUIRE(dag.build());
    REQUIRE(dag.execute());
    REQUIRE(dag.execution_order().size() == 2);
    REQUIRE(std::any_cast<int>(dag.get_node("d2")->get_output("out")) == 12);
}

TEST_CASE("Dag connect port/type validation", "[pipeline]") {
    Dag dag;
    // 不存在节点 / 不存在端口 → connect false
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    REQUIRE_FALSE(dag.connect("nope", "out", "b", "in"));
    REQUIRE_FALSE(dag.connect("a", "nope", "b", "in"));
    // 类型不匹配：a.out(int) -> img_in.in(Image) → connect false
    dag.add_node(std::make_unique<ImageSinkNode>("sink"));
    REQUIRE_FALSE(dag.connect("a", "out", "sink", "in"));
}

TEST_CASE("Dag cycle detected by build", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    REQUIRE(dag.connect("a", "out", "b", "in"));
    REQUIRE(dag.connect("b", "out", "a", "in"));
    REQUIRE_FALSE(dag.build());
}

TEST_CASE("Dag build fails on unconnected required input", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    // b 未连边也未种子 → build false
    REQUIRE_FALSE(dag.build());
}

TEST_CASE("Dag fan-out executes once each", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("c"));
    dag.add_node(std::make_unique<DoubleNode>("d"));
    REQUIRE(dag.connect("a", "out", "c", "in"));
    REQUIRE(dag.connect("a", "out", "d", "in"));
    dag.get_node("a")->set_input("in", std::any(5));
    REQUIRE(dag.build());
    REQUIRE(dag.execute());
    REQUIRE(std::any_cast<int>(dag.get_node("c")->get_output("out")) == 10);
    REQUIRE(std::any_cast<int>(dag.get_node("d")->get_output("out")) == 10);
}

TEST_CASE("Dag exec order given by topo sort", "[pipeline]") {
    Dag dag;
    dag.add_node(std::make_unique<DoubleNode>("a"));
    dag.add_node(std::make_unique<DoubleNode>("b"));
    dag.add_node(std::make_unique<DoubleNode>("c"));
    REQUIRE(dag.connect("a", "out", "b", "in"));
    REQUIRE(dag.connect("a", "out", "c", "in"));
    dag.get_node("a")->set_input("in", std::any(2));
    REQUIRE(dag.build());
    auto order = dag.execution_order();
    REQUIRE(order.size() == 3);
    // a 必须在 b、c 之前
    auto ia = std::find(order.begin(), order.end(), "a");
    auto ib = std::find(order.begin(), order.end(), "b");
    auto ic = std::find(order.begin(), order.end(), "c");
    REQUIRE(ia < ib);
    REQUIRE(ia < ic);
    REQUIRE(dag.execute());
}

- [ ] **Step 2: Run to verify fail**

Run: `cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: FAIL（`csrc/pipeline/dag.h` 不存在，编译失败）

- [ ] **Step 3: Write `csrc/pipeline/dag.h`**

```cpp
#pragma once

#include "core/md_decl.h"
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/edge.h"
#include <memory>
#include <string>
#include <vector>

namespace modeldeploy::pipeline {

// 有向无环图：注册节点 + 边，build() 校验并拓扑排序，execute() 拓扑序单线程执行。
class MODELDEPLOY_CXX_EXPORT Dag {
public:
    void add_node(std::unique_ptr<Node> node);
    Node* get_node(const std::string& name);

    // src_node.out_port -> dst_node.in_port；校验节点/端口存在且类型一致
    bool connect(const std::string& src_node, const std::string& src_port,
                 const std::string& dst_node, const std::string& dst_port);

    // 校验：无环 + 每个声明输入端口已连边或已种子 → 生成拓扑序；失败返回 false
    bool build();
    // 按拓扑序执行：前置清空各节点输出；每节点 run() 后把输出拷到下游输入
    bool execute();

    std::vector<std::string> execution_order() const { return order_; }
    const std::vector<Edge>& edges() const { return edges_; }

private:
    Node* find_node(const std::string& name);

    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Edge> edges_;
    std::vector<std::string> order_;
    bool built_ = false;
};

} // namespace modeldeploy::pipeline
```

- [ ] **Step 4: Write `csrc/pipeline/dag.cpp`**

```cpp
#include "csrc/pipeline/dag.h"
#include <deque>
#include <unordered_map>

namespace modeldeploy::pipeline {

Node* Dag::find_node(const std::string& name) {
    for (auto& n : nodes_) {
        if (n->name() == name) return n.get();
    }
    return nullptr;
}

void Dag::add_node(std::unique_ptr<Node> node) {
    nodes_.push_back(std::move(node));
    built_ = false;
}

Node* Dag::get_node(const std::string& name) { return find_node(name); }

bool Dag::connect(const std::string& src_node, const std::string& src_port,
                  const std::string& dst_node, const std::string& dst_port) {
    Node* s = find_node(src_node);
    Node* d = find_node(dst_node);
    if (!s || !d) return false;
    const Port* sp = nullptr;
    const Port* dp = nullptr;
    for (const auto& p : s->outputs()) if (p.name == src_port) { sp = &p; break; }
    for (const auto& p : d->inputs())  if (p.name == dst_port) { dp = &p; break; }
    if (!sp || !dp) return false;
    if (sp->type != dp->type) return false;  // 端口逻辑类型必须一致
    edges_.push_back(Edge{src_node, src_port, sp->type, dst_node, dst_port, dp->type});
    built_ = false;
    return true;
}

bool Dag::build() {
    if (nodes_.empty()) { built_ = true; order_.clear(); return true; }

    // 校验：每个声明的输入端口必须已连边 或 已种子（has_input）
    for (const auto& n : nodes_) {
        for (const auto& p : n->inputs()) {
            bool connected = false;
            for (const auto& e : edges_)
                if (e.dst_node == n->name() && e.dst_port == p.name) { connected = true; break; }
            if (!connected && !n->has_input(p.name)) return false;
        }
    }

    // Kahn 拓扑排序
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int> indeg;
    for (const auto& n : nodes_) { adj[n->name()] = {}; indeg[n->name()] = 0; }
    for (const auto& e : edges_) {
        adj[e.src_node].push_back(e.dst_node);
        indeg[e.dst_node]++;
    }
    std::deque<std::string> zero;
    for (const auto& n : nodes_) if (indeg[n->name()] == 0) zero.push_back(n->name());
    order_.clear();
    while (!zero.empty()) {
        std::string name = zero.front();
        zero.pop_front();
        order_.push_back(name);
        for (const auto& succ : adj[name])
            if (--indeg[succ] == 0) zero.push_back(succ);
    }
    built_ = (order_.size() == nodes_.size());  // 有环则 order 不全
    return built_;
}

bool Dag::execute() {
    if (!built_) return false;
    for (const auto& n : nodes_) n->clear_outputs();
    for (const auto& name : order_) {
        Node* n = find_node(name);
        if (!n) return false;
        if (!n->run()) return false;
        for (const auto& e : edges_) {
            if (e.src_node != name) continue;
            Node* d = find_node(e.dst_node);
            if (!d) return false;
            d->set_input(e.dst_port, n->get_output(e.src_port));
        }
    }
    return true;
}

} // namespace modeldeploy::pipeline
```

- [ ] **Step 5: Run to verify pass**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: 全部 `[pipeline]` 用例 PASS（含 Task 1 的 3 个）。

- [ ] **Step 6: Commit**

```bash
git add csrc/pipeline/dag.h csrc/pipeline/dag.cpp tests/test_pipeline.cpp
git commit -m "feat(pipeline): Dag topo-sort, cycle/unconnected validation, execute dataflow"
```

---

### Task 3: `Planner`（最小 DSL：顺序 + 单级 fan-in/fan-out）

**Files:**
- Create: `csrc/pipeline/planner.h`
- Create: `csrc/pipeline/planner.cpp`
- Modify: `tests/test_pipeline.cpp`

**Interfaces:**
- Consumes: Node/Edge/NodeFactory (Task 1/2)
- Produces (later tasks rely on):
  - `class Planner::using Factory = std::function<std::unique_ptr<Node>(const std::string& instance)>`
  - `void Planner::register_model(const std::string& name, Factory f)`
  - `std::unique_ptr<Dag> Planner::build(const std::string& spec)`
  - DSL 语义：`A -> B -> C`（顺序）、`{A,B} -> C`（fan-in，目标节点输入端口为 `in`,`in1`,`in2`,...）、`A -> {B,C}`（fan-out，源 `out` → 各目标 `in`）。多对多不支持（返回 nullptr）。
  - 实例名唯一性：同一 model 名在多段重复时 Planner 自动加 `_N` 后缀（首个不后缀）。

- [ ] **Step 1: Append failing tests** (`tests/test_pipeline.cpp`)

```cpp
#include "csrc/pipeline/planner.h"
#include <functional>

TEST_CASE("Planner sequential DSL", "[pipeline]") {
    Planner p;
    p.register_model("d", [](const std::string& inst) { return std::make_unique<DoubleNode>(inst); });
    auto dag = p.build("d -> d");   // 实例名 d, d_1
    REQUIRE(dag != nullptr);
    dag->get_node("d")->set_input("in", std::any(3));
    REQUIRE(dag->build());
    REQUIRE(dag->execute());
    REQUIRE(dag->execution_order().size() == 2);
    REQUIRE(std::any_cast<int>(dag->get_node("d_1")->get_output("out")) == 12);
}

TEST_CASE("Planner fan-out DSL", "[pipeline]") {
    Planner p;
    p.register_model("d", [](const std::string& inst) { return std::make_unique<DoubleNode>(inst); });
    auto dag = p.build("d -> {d, d}");   // d, d_1, d_2
    REQUIRE(dag != nullptr);
    dag->get_node("d")->set_input("in", std::any(5));
    REQUIRE(dag->build());
    REQUIRE(dag->execute());
    REQUIRE(dag->execution_order().size() == 3);
    REQUIRE(std::any_cast<int>(dag->get_node("d_1")->get_output("out")) == 10);
    REQUIRE(std::any_cast<int>(dag->get_node("d_2")->get_output("out")) == 10);
}

TEST_CASE("Planner fan-in DSL", "[pipeline]") {
    Planner p;
    p.register_model("id", [](const std::string& inst) { return std::make_unique<IdentityNode>(inst); });
    // AddNode: in + in1 两个 int 输入，输出和
    struct AddNode : Node {
        AddNode(std::string n) : Node(std::move(n), {{"in", "int"}, {"in1", "int"}}, {{"out", "int"}}) {}
        bool run() override {
            int a, b;
            if (!get_in<int>("in", &a) || !get_in<int>("in1", &b)) return false;
            set_out<int>("out", a + b);
            return true;
        }
    };
    p.register_model("add", [](const std::string& inst) { return std::make_unique<AddNode>(inst); });
    auto dag = p.build("{id, id} -> add");   // id, id_1, add
    REQUIRE(dag != nullptr);
    dag->get_node("id")->set_input("in", std::any(1));
    dag->get_node("id_1")->set_input("in", std::any(2));
    REQUIRE(dag->build());
    REQUIRE(dag->execute());
    REQUIRE(dag->execution_order().size() == 3);
    REQUIRE(std::any_cast<int>(dag->get_node("add")->get_output("out")) == 3);
}

TEST_CASE("Planner unknown model returns nullptr", "[pipeline]") {
    Planner p;
    REQUIRE(p.build("ghost -> d") == nullptr);
}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: FAIL（`csrc/pipeline/planner.h` 不存在，编译失败）

- [ ] **Step 3: Write `csrc/pipeline/planner.h`**

```cpp
#pragma once

#include "core/md_decl.h"
#include "csrc/pipeline/dag.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace modeldeploy::pipeline {

// 根据最小 DSL 自动布节点/边。YAGNI：只支持顺序 + 单级 fan-in/fan-out。
// 复杂图直接用 Dag::add_node/connect 手工搭。
class MODELDEPLOY_CXX_EXPORT Planner {
public:
    // 工厂：给定实例名创建 Node（端口 schema 由实现自行声明）
    using Factory = std::function<std::unique_ptr<Node>(const std::string& instance)>;

    void register_model(const std::string& name, Factory f);
    std::unique_ptr<Dag> build(const std::string& spec);

private:
    std::unordered_map<std::string, Factory> factories_;
};

} // namespace modeldeploy::pipeline
```

- [ ] **Step 4: Write `csrc/pipeline/planner.cpp`**

```cpp
#include "csrc/pipeline/planner.h"
#include <cctype>

namespace modeldeploy::pipeline {

void Planner::register_model(const std::string& name, Factory f) {
    factories_[name] = std::move(f);
}

namespace {
    std::string trim(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
        return s.substr(b, e - b);
    }

    // 拆顶层 "->"，得到各段
    std::vector<std::string> split_stages(const std::string& spec) {
        std::vector<std::string> out;
        size_t pos = 0;
        while (true) {
            auto arrow = spec.find("->", pos);
            if (arrow == std::string::npos) { out.push_back(trim(spec.substr(pos))); break; }
            out.push_back(trim(spec.substr(pos, arrow - pos)));
            pos = arrow + 2;
        }
        return out;
    }

    // 解析单段：可含单个名或 {A, B} 组
    std::vector<std::string> parse_stage(const std::string& seg) {
        std::vector<std::string> out;
        if (seg.empty()) return out;
        if (seg.front() == '{') {
            std::string cur;
            for (char c : seg) {
                if (c == '{' || c == '}') continue;
                if (c == ',') { out.push_back(trim(cur)); cur.clear(); }
                else cur += c;
            }
            if (!trim(cur).empty()) out.push_back(trim(cur));
        } else {
            out.push_back(seg);
        }
        return out;
    }

    // 把 model 名映射为唯一实例名（重复加 _N）
    std::string unique_instance(const std::string& model,
                                std::unordered_map<std::string, int>& counts) {
        int k = counts[model]++;
        return (k == 0) ? model : (model + "_" + std::to_string(k));
    }
} // namespace

std::unique_ptr<Dag> Planner::build(const std::string& spec) {
    std::vector<std::string> stages = split_stages(spec);
    if (stages.empty()) return nullptr;

    auto dag = std::make_unique<Dag>();
    std::unordered_map<std::string, int> counts;
    std::vector<std::vector<std::string>> groups;  // 每段的实例名

    for (const auto& st : stages) {
        std::vector<std::string> g = parse_stage(st);
        std::vector<std::string> instances;
        for (const auto& name : g) {
            auto it = factories_.find(name);
            if (it == factories_.end()) return nullptr;  // 未注册模型
            std::string inst = unique_instance(name, counts);
            dag->add_node(it->second(inst));
            instances.push_back(inst);
        }
        groups.push_back(std::move(instances));
    }

    for (size_t i = 0; i + 1 < groups.size(); ++i) {
        const auto& from = groups[i];
        const auto& to = groups[i + 1];
        if (from.size() == 1 && to.size() == 1) {
            if (!dag->connect(from[0], "out", to[0], "in")) return nullptr;
        } else if (from.size() > 1 && to.size() == 1) {
            // fan-in：源 k 连目标 in（k=0 为 in，k>=1 为 inN）
            for (size_t k = 0; k < from.size(); ++k) {
                std::string inport = (k == 0) ? std::string("in") : ("in" + std::to_string(k));
                if (!dag->connect(from[k], "out", to[0], inport)) return nullptr;
            }
        } else if (from.size() == 1 && to.size() > 1) {
            // fan-out：源 out -> 各目标 in
            for (const auto& t : to) {
                if (!dag->connect(from[0], "out", t, "in")) return nullptr;
            }
        } else {
            return nullptr;  // 多对多不支持（YAGNI）
        }
    }

    dag->build();  // 让 build 校验，调用方可再显式 build()
    return dag;
}

} // namespace modeldeploy::pipeline
```

> 注：`build()` 末尾调用 `dag->build()` 仅为触发校验置 `built_`；测试中仍显式 `dag->build()` 以确认返回 true。

- [ ] **Step 5: Run to verify pass**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[pipeline]"`
Expected: 全部 `[pipeline]` 用例 PASS（含顺序/fan-out/fan-in/未知模型）。

- [ ] **Step 6: Commit**

```bash
git add csrc/pipeline/planner.h csrc/pipeline/planner.cpp tests/test_pipeline.cpp
git commit -m "feat(pipeline): Planner minimal DSL (sequential + single-level fan in/out)"
```

---

### Task 4: Python 绑定（`pipeline` 子模块）

**Files:**
- Create: `csrc/pybind/pipeline/pipeline_pybind.cpp`
- Modify: `csrc/pybind/main.cpp`
- Test: （Python smoke）

**Interfaces:**
- Consumes: Node/Planner (Task 1-3)
- Produces: Python `modeldeploy.pipeline.Dag` / `Node`（`get_node` 结果）/ `Planner`（`Planner.register_transform(name, type, fn)` + `build(spec)` → `Dag`）

- [ ] **Step 1: Write `csrc/pybind/pipeline/pipeline_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <functional>
#include <any>
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/dag.h"
#include "csrc/pipeline/planner.h"

namespace py = pybind11;
using namespace modeldeploy::pipeline;

namespace modeldeploy::pipeline {

namespace {
    // Python 友好的 TransformNode：包装一个 py 可调用；in/out 端口 type 为注册时传入的 type 字符串
    class PyTransformNode : public Node {
    public:
        PyTransformNode(std::string name, std::string type, py::object fn)
            : Node(std::move(name), {{"in", type}}, {{"out", type}}),
              fn_(std::move(fn)) {}
        bool run() override {
            py::object v;
            if (!get_in<py::object>("in", &v)) return false;
            py::object r = fn_(v);
            set_out<py::object>("out", r);
            return true;
        }
    private:
        py::object fn_;
    };
} // namespace

void bind_pipeline(pybind11::module& m) {
    // Node：供 Dag.get_node 结果读写数据；以 py::object 承载
    py::class_<Node>(m, "Node")
        .def("name", &Node::name)
        .def("set_input", [](Node& n, const std::string& p, py::object v) {
            n.set_input(p, std::any(py::object(v)));
        })
        .def("get_output", [](Node& n, const std::string& p) -> py::object {
            std::any a = n.get_output(p);
            if (!a.has_value()) return py::none();
            return py::any_cast<py::object>(a);
        });

    // Dag：节点由 Planner 建立；Python 侧主要 get_node + build + execute
    py::class_<Dag>(m, "Dag")
        .def(py::init<>())
        .def("get_node", &Dag::get_node, py::return_value_policy::reference)
        .def("connect", &Dag::connect)
        .def("build", &Dag::build)
        .def("execute", &Dag::execute)
        .def("execution_order", &Dag::execution_order);

    // Planner：注册 Python 变换函数，据 DSL 布图返回 Dag
    py::class_<Planner>(m, "Planner")
        .def(py::init<>())
        .def("register_transform", [](Planner& p, const std::string& name,
                                      const std::string& type, py::object fn) {
            p.register_model(name, [fn = py::object(fn), type](const std::string& inst) {
                return std::unique_ptr<Node>(new PyTransformNode(inst, type, fn));
            });
        })
        .def("build", [](Planner& p, const std::string& spec) { return p.build(spec); });

    m.attr("__doc__") = "Pipeline DAG orchestration module.";
}

} // namespace modeldeploy::pipeline
```

> **实现说明**：不暴露 Python 侧 `Node` 子类化 / `Dag.add_node(Node)`（避免 pybind holder/多态转换负担，YAGNI）。DAG 内节点由 `Planner.register_transform` 在 C++ 侧构建并**归 Dag 所有**（`std::unique_ptr<Node>` 默认删除器，生命期安全）；Python 通过 `dag.get_node(name)` 拿引用做 `set_input`/`get_output`。`std::any` 内实际包装 `py::object`，Dag 拷贝 any 即拷贝 `py::object`（引用计数+1）。lambda 均在 Python 驱动的调用（register/build/execute）中执行，pybind 持有 GIL，引用计数安全。

- [ ] **Step 2: Register in `csrc/pybind/main.cpp`**

修改 main.cpp：加声明 + 注册子模块（pipeline 无 BUILD 条件，纯 C++ 常编译）：

```cpp
namespace modeldeploy::pipeline {
    void bind_pipeline(pybind11::module&);
}
```
（在 `namespace modeldeploy {` 内 PYBIND11_MODULE 中，`bind_base_model(m);` 之后追加）
```cpp
        auto pipeline_module =
            m.def_submodule("pipeline", "Pipeline DAG module of Modeldeploy.");
        pipeline::bind_pipeline(pipeline_module);
```

- [ ] **Step 3: Build + Python smoke**

用开 `BUILD_PYTHON=ON` 的构建（例如 `build_py`）。构建后：
```bash
cd build_py && python -c "
import modeldeploy.pipeline as p

# 顺序 DSL: d -> d (实例 d, d_1)
pl = p.Planner()
pl.register_transform('d','int', lambda v: v*2)
dag = pl.build('d -> d')
dag.get_node('d').set_input('in', 3)
assert dag.build() and dag.execute()
assert dag.get_node('d_1').get_output('out') == 12

# 多段不同变换: a -> add -> b
pl2 = p.Planner()
pl2.register_transform('a','int', lambda v: v+10)
pl2.register_transform('add','int', lambda v: v)
pl2.register_transform('b','int', lambda v: v*3)
dag2 = pl2.build('a -> add -> b')
dag2.get_node('a').set_input('in', 5)   # (5+10)=15 ->15*3=45
assert dag2.build() and dag2.execute()
assert dag2.get_node('b').get_output('out') == 45

# 未知模型返回 None
assert pl2.build('ghost -> b') is None
print('pipeline smoke OK')
"
```
Expected: `pipeline smoke OK`，无异常。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/pipeline/pipeline_pybind.cpp csrc/pybind/main.cpp
git commit -m "feat(pybind): bind pipeline Dag + Planner (register_transform/build)"
```

---

## 子系统 B：视频软解下沉 SDK（FFmpeg，BUILD_VIDEO 默认 OFF）

### Task 5: `cmake/ffmpeg.cmake` + 根 CMake `BUILD_VIDEO` 接线

**Files:**
- Create: `cmake/ffmpeg.cmake`
- Modify: `CMakeLists.txt`（option、VIDEO_SOURCE glob、REMOVE_ITEM、BUILD_VIDEO 块）
- Modify: `tests/CMakeLists.txt`（条件追加 test_video_decoder.cpp）

**Interfaces:**
- Produces (Task 6/7 rely on):
  - CMake 变量 `FFMPEG_INCLUDE_DIR`、`FFMPEG_LIBS`；`BUILD_VIDEO` 选项；`csrc/video/*.cpp` 仅 BUILD_VIDEO 时编译。
  - 若 FFmpeg 找不到，ffmpeg.cmake 把 `BUILD_VIDEO` 置 OFF 并 WARNING（优雅禁用）。

- [ ] **Step 1: Write `cmake/ffmpeg.cmake`**

```cmake
# FFmpeg 查找模块：产出 FFMPEG_INCLUDE_DIR 与 FFMPEG_LIBS（模块化 application/CMakeLists.txt 的 find 逻辑）。
# 用法：option(BUILD_VIDEO ...) 为 ON 时 include 本文件。
# 找不到 FFmpeg 时把 BUILD_VIDEO 置 OFF（优雅禁用），由上层据此跳过视频编译。

set(FFMPEG_ROOT "E:/develop/ffmpeg" CACHE PATH "FFmpeg installation root")
find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h PATHS ${FFMPEG_ROOT}/include NO_DEFAULT_PATH)

set(FFMPEG_LIBS "")
foreach (lib avcodec avformat avutil swscale avdevice)
    find_library(FFMPEG_${lib}_LIBRARY NAMES ${lib} ${lib}.lib
            PATHS ${FFMPEG_ROOT}/lib NO_DEFAULT_PATH)
    if (FFMPEG_${lib}_LIBRARY)
        list(APPEND FFMPEG_LIBS ${FFMPEG_${lib}_LIBRARY})
    endif ()
endforeach ()

if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)
    message(WARNING "FFmpeg not found at ${FFMPEG_ROOT}; BUILD_VIDEO disabled")
    set(BUILD_VIDEO OFF)
endif ()
```

- [ ] **Step 2: Modify root `CMakeLists.txt`**

(1) 在 option 区（`BUILD_SURVEILLANCE` 后，约 23 行）加：
```cmake
option(BUILD_VIDEO "build video decoder module (requires FFmpeg and BUILD_VISION)" OFF)
```

(2) 在 `file(GLOB_RECURSE ...)` 区（约 116-118）加 VIDEO_SOURCE：
```cmake
file(GLOB_RECURSE VIDEO_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/video/*.cpp)
```

(3) 第 124 行 REMOVE_ITEM 追加 `${VIDEO_SOURCE}`（**关键**：否则 BUILD_VIDEO=OFF 也编译 csrc/video）：

```cmake
list(REMOVE_ITEM ALL_SOURCE ${ORT_BACKEND_SOURCE} ${MNN_BACKEND_SOURCE} ${TRT_BACKEND_SOURCE} ${SOPHGO_BACKEND_SOURCE} ${VISION_SOURCE} ${AUDIO_SOURCE} ${PYBIND_SOURCE} ${CUDA_SOURCE} ${VIDEO_SOURCE})
```

(4) 在 `BUILD_AUDIO` 块（219 行）之后、`BUILD_BARCODE` 块之前插入 VIDEO 块：

```cmake
# ── VIDEO ──────────────────────────────────
if (BUILD_VIDEO)
    if (NOT BUILD_VISION)
        message(FATAL_ERROR "BUILD_VIDEO requires BUILD_VISION=ON (VideoDecoder returns vision::ImageData)")
    endif ()
    include("${CMAKE_SOURCE_DIR}/cmake/ffmpeg.cmake")
endif ()
if (BUILD_VIDEO)
    add_definitions(-DBUILD_VIDEO)
    list(APPEND ALL_SOURCE ${VIDEO_SOURCE})
    list(APPEND DEPENDS ${FFMPEG_LIBS})
endif ()
```

- [ ] **Step 3: Modify `tests/CMakeLists.txt`**

在 `endwith` 处（`add_executable(test_modeldeploy ...)` 之后、`add_test` 之前）追加：
```cmake
# 视频解码测试（仅 BUILD_VIDEO）
if (BUILD_VIDEO)
    list(APPEND TEST_SOURCES test_video_decoder.cpp)
    target_include_directories(test_modeldeploy PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_libraries(test_modeldeploy PRIVATE ${FFMPEG_LIBS})
endif ()
```

- [ ] **Step 4: Verify configure without FFmpeg**（优雅禁用分支 —— `csrc/video/` 尚无文件，主要验证配置不报错）

Run: `cmake -S . -B build_cpu -G Ninja -DBUILD_VIDEO=ON -DBUILD_VISION=ON -DBUILD_TESTS=ON -DENABLE_ORT=ON -DENABLE_MNN=OFF -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_AUDIO=OFF -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF`
Expected: 配置成功；`FFmpeg not found` WARNING 或 `BUILD_VIDEO` 生效（本机 `E:/develop/ffmpeg` 存在则视频启用）。
> 本任务不涉及源码，`csrc/video/` 尚空，BUILD_VIDEO=ON 时 GLOB 收集空集，编译仍通过。

- [ ] **Step 5: Commit**

```bash
git add cmake/ffmpeg.cmake CMakeLists.txt tests/CMakeLists.txt
git commit -m "build(video): modular ffmpeg.cmake + BUILD_VIDEO option (OFF by default)"
```

---

### Task 6: `video::VideoDecoder`（FFmpeg 软解核心下沉）+ 视频测试

**Files:**
- Create: `csrc/video/video_decoder.h`
- Create: `csrc/video/video_decoder.cpp`
- Test: `tests/test_video_decoder.cpp`

**Interfaces:**
- Consumes: `vision::ImageData::from_planes`（`image_data.h:75`）、`MdImageType::NV12`、`Device::CPU`（Task 5 的 BUILD_VIDEO/FFmpeg 变量）
- Produces (Task 7/8 rely on):
  - `modeldeploy::video::VideoDecoder()`；`bool open(const std::string& url)`；`bool next(ImageData* out, uint64_t* pts_ms)`；`void close()`；`int width() const`；`int height() const`；`double fps() const`。
  - 软解输出 CPU NV12 `ImageData`（owner 持有解码 AVFrame 生命周期）。

- [ ] **Step 1: Write failing test** (`tests/test_video_decoder.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <sys/stat.h>
#include "csrc/video/video_decoder.h"

using namespace modeldeploy::video;
using namespace modeldeploy::vision;

static bool file_exists(const char* p) {
    if (!p || !*p) return false;
    struct stat st;
    return ::stat(p, &st) == 0;
}

TEST_CASE("VideoDecoder opens and grabs NV12 frames", "[video]") {
    const char* p = std::getenv("MD_TEST_VIDEO");
    if (!file_exists(p)) {
        SKIP("no test video; set MD_TEST_VIDEO to a local mp4");
    }
    VideoDecoder dec;
    REQUIRE(dec.open(p));
    REQUIRE(dec.width() > 0);
    REQUIRE(dec.height() > 0);
    REQUIRE(dec.fps() > 0.0);

    ImageData frame;
    uint64_t pts = 0;
    int got = 0;
    while (got < 5 && dec.next(&frame, &pts)) {
        REQUIRE_FALSE(frame.empty());
        REQUIRE(frame.width() == dec.width());
        REQUIRE(frame.height() == dec.height());
        ++got;
    }
    REQUIRE(got > 0);           // 至少抽到一帧
    REQUIRE(frame.plane_count() >= 1);
    dec.close();
}
```

- [ ] **Step 2: Run to verify fail**

Run（带 BUILD_VIDEO + TESTS 的构建）: `cmake --build build_video --parallel 8 && cd build_video && $env:MD_TEST_VIDEO="path\to\test.mp4" ; .\bin\test_modeldeploy.exe "[video]"`
Expected: FAIL 或编译失败（`csrc/video/video_decoder.h` 不存在）。

- [ ] **Step 3: Write `csrc/video/video_decoder.h`**

```cpp
#pragma once

#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include <cstdint>
#include <string>

// FFmpeg C 头
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

namespace modeldeploy::video {

// SDK 视频软解：复用 application/stream_decoder 的 FFmpeg 软解核心（h264/hevc 软解 → NV12）。
// YAGNI：只做 FFmpeg 软解，不做 CUVID/Sophgo/hw_accel（避免 SDK 引入 CUDA 依赖）。
class MODELDEPLOY_CXX_EXPORT VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder();

    bool open(const std::string& url);
    // 抽下一帧；成功时 *out 为 CPU NV12 ImageData（owner 保活解码帧），*pts_ms 为毫秒时间戳
    bool next(ImageData* out, uint64_t* pts_ms);
    void close();

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }

private:
    void cleanup();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* dec_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;    // 解码器输出帧（被 next 内 ref 到 owner）
    int width_ = 0;
    int height_ = 0;
    double fps_ = 25.0;
};

} // namespace modeldeploy::video
```

- [ ] **Step 4: Write `csrc/video/video_decoder.cpp`**

```cpp
#include "csrc/video/video_decoder.h"
#include <memory>

extern "C" {
#include <libavutil/imgutils.h>
}

namespace modeldeploy::video {

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open(const std::string& url) {
    cleanup();
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "stimeout", "5000000", 0);
    fmt_ctx_ = avformat_alloc_context();
    if (avformat_open_input(&fmt_ctx_, url.c_str(), nullptr, &opts) < 0) {
        av_dict_free(&opts);
        cleanup();
        return false;
    }
    av_dict_free(&opts);
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        cleanup();
        return false;
    }
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        auto* cp = fmt_ctx_->streams[i]->codecpar;
        if (cp->codec_type != AVMEDIA_TYPE_VIDEO) continue;
        const AVCodec* dec = avcodec_find_decoder(cp->codec_id);   // 仅软解
        if (!dec) { cleanup(); return false; }
        video_stream_idx_ = static_cast<int>(i);
        dec_ctx_ = avcodec_alloc_context3(dec);
        avcodec_parameters_to_context(dec_ctx_, cp);
        if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) { cleanup(); return false; }
        width_ = cp->width;
        height_ = cp->height;
        auto* st = fmt_ctx_->streams[i];
        if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)
            fps_ = static_cast<double>(st->avg_frame_rate.num) / st->avg_frame_rate.den;
        if (fps_ <= 0.0) fps_ = 25.0;
        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        return true;
    }
    cleanup();
    return false;
}

bool VideoDecoder::next(ImageData* out, uint64_t* pts_ms) {
    if (!fmt_ctx_ || !dec_ctx_ || !out) return false;
    while (true) {
        int ret = avcodec_receive_frame(dec_ctx_, frame_);
        if (ret == 0) {
            // 把解码帧 ref 到自有 owner，保证 ImageData 生命周期内缓冲有效
            std::shared_ptr<AVFrame> owned(av_frame_alloc(),
                                           [](AVFrame* f) { av_frame_free(&f); });
            if (av_frame_ref(owned.get(), frame_) < 0) return false;
            if (!owned->data[0] || !owned->data[1]) return false;  // 只处理 NV12 双平面
            ImageData::Plane pl[2] = {
                {owned->data[0], owned->linesize[0]},
                {owned->data[1], owned->linesize[1]},
            };
            *out = ImageData::from_planes(pl, 2, modeldeploy::vision::MdImageType::NV12,
                                          owned->width, owned->height,
                                          modeldeploy::Device::CPU, owned);
            if (pts_ms) {
                auto* st = fmt_ctx_->streams[video_stream_idx_];
                *pts_ms = (owned->pts == AV_NOPTS_VALUE)
                              ? 0
                              : static_cast<uint64_t>(
                                    av_rescale_q(owned->pts, st->time_base, AVRational{1, 1000}));
            }
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            av_packet_unref(pkt_);
            int r = av_read_frame(fmt_ctx_, pkt_);
            if (r < 0) return false;  // EOF/错误（本地文件正常结束）
            if (pkt_->stream_index != video_stream_idx_) { av_packet_unref(pkt_); continue; }
            if (avcodec_send_packet(dec_ctx_, pkt_) < 0) { av_packet_unref(pkt_); continue; }
            continue;
        }
        return false;
    }
}

void VideoDecoder::close() { cleanup(); }

void VideoDecoder::cleanup() {
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (dec_ctx_) avcodec_free_context(&dec_ctx_);
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_idx_ = -1;
    width_ = 0;
    height_ = 0;
}

} // namespace modeldeploy::video
```

- [ ] **Step 5: Register test already done in Task 5 (tests/CMakeLists.txt); build + run**

Run: `cmake --build build_video --parallel 8`
Expected: 0 errors（`csrc/video/*.cpp` 编译并链入 SDK + FFMPEG_LIBS）。

运行视频测试（设置 `MD_TEST_VIDEO` 指向本地 mp4，例如 test_data 中的某个文件；FFmpeg DLL 需在 PATH）：
```powershell
cd build_video
$env:PATH = "E:/develop/ffmpeg/bin;$env:PATH"
$env:MD_TEST_VIDEO = "E:/path/to/sample.mp4"
.\bin\test_modeldeploy.exe "[video]"
```
Expected: 若文件存在 → PASS（宽/高/fps/断言 5 帧）；否则 `SKIP`（显示 `skipped`）。

- [ ] **Step 6: Commit**

```bash
git add csrc/video/video_decoder.h csrc/video/video_decoder.cpp tests/test_video_decoder.cpp tests/CMakeLists.txt CMakeLists.txt cmake/ffmpeg.cmake
git commit -m "feat(video): VideoDecoder FFmpeg soft-decode to NV12 ImageData + [video] test"
```

---

### Task 7: Python 绑定（`video` 子模块）

**Files:**
- Create: `csrc/pybind/video/video_pybind.cpp`
- Modify: `csrc/pybind/main.cpp`
- Test: （Python smoke）

**Interfaces:**
- Consumes: VideoDecoder (Task 6)
- Produces: Python `modeldeploy.video.VideoDecoder(url)`（`open()/next()/close()/width()/height()/fps()`，`next()` 返回 `vision.ImageData`）

- [ ] **Step 1: Write `csrc/pybind/video/video_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>

// 仅 BUILD_VIDEO && BUILD_VISION 时才注册（ImageData 绑定来自 vision 子模块）
#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
#include "csrc/video/video_decoder.h"

namespace py = pybind11;
using namespace modeldeploy::video;
using namespace modeldeploy::vision;

namespace modeldeploy::video {
void bind_video(pybind11::module& m) {
    py::class_<VideoDecoder>(m, "VideoDecoder")
        .def(py::init<>())
        .def("open", &VideoDecoder::open, py::arg("url"))
        .def("next", [](VideoDecoder& d) {
            ImageData f;
            uint64_t pts = 0;
            bool ok = d.next(&f, &pts);
            return std::make_pair(ok, f);   // ImageData 已注册，返回 vision.ImageData
        })
        .def("close", &VideoDecoder::close)
        .def_property_readonly("width", &VideoDecoder::width)
        .def_property_readonly("height", &VideoDecoder::height)
        .def_property_readonly("fps", &VideoDecoder::fps);
}
} // namespace modeldeploy::video
#endif
```

> `next()` 返回 `(bool, ImageData)`。`ImageData` 同一扩展模块内已注册（`vision.ImageData`），pybind 类型注册表共享，自动复用其绑定。

- [ ] **Step 2: Register in `csrc/pybind/main.cpp`**

追加声明（`namespace modeldeploy::video { ... }`）与注册（放在 pipeline 子模块后）：
```cpp
#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
namespace modeldeploy::video { void bind_video(pybind11::module&); }
#endif
```
在 PYBIND11_MODULE 内：
```cpp
#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
        auto video_module =
            m.def_submodule("video", "Video decode module of Modeldeploy.");
        video::bind_video(video_module);
#endif
```

- [ ] **Step 3: Build + Python smoke**

在开 `BUILD_VIDEO=ON BUILD_VISION=ON BUILD_PYTHON=ON` 的构建（`build_video_py`）后：
```bash
cd build_video_py && python -c "
import modeldeploy.video as v
import os
url = os.environ.get('MD_TEST_VIDEO','')
dec = v.VideoDecoder()
assert dec.open(url)
print('size', dec.width, 'x', dec.height, 'fps', dec.fps)
ok, frame = dec.next()
print('next', ok, 'empty', frame.empty() if hasattr(frame,'empty') else 'n/a')
dec.close()
"
```
Expected: 打开本地 mp4 并打印宽高/fps；`ok=True`。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/video/video_pybind.cpp csrc/pybind/main.cpp
git commit -m "feat(pybind): bind video.VideoDecoder under BUILD_VIDEO"
```

---

### Task 8（可选 / 可降级跳过）：CAPI 薄封装

> **YAGNI**：DAG/视频在 CAPI 只做能代表使用的薄封装；若时间紧可跳过本任务（C++/Python 已为主要交付）。

**Files:**
- Modify: `capi/md_capi.h`
- Modify: `capi/md_capi.cpp`

**Interfaces:**
- Consumes: Planner/VideoDecoder (Task 3/6)
- Produces: `md_pipeline_create/build/execute/destroy`（DSL 字符串建图）+ `md_video_decoder_create/next/destroy`（`#ifdef BUILD_VIDEO`）。底层复用同一 `Dag`/`VideoDecoder`；不暴露任意 Node 图构造。

- [ ] **Step 1**: `md_capi.h` 声明
  ```c
  typedef void* MDPipelineHandle;
  MDPipelineHandle md_pipeline_create(const char* spec);
  int md_pipeline_build(MDPipelineHandle h);
  int md_pipeline_execute(MDPipelineHandle h);
  void md_pipeline_destroy(MDPipelineHandle h);
  #ifdef BUILD_VIDEO
  typedef void* MDVideoDecoderHandle;
  MDVideoDecoderHandle md_video_decoder_create(const char* url);
  int md_video_decoder_next(MDVideoDecoderHandle h, MDImageHandle* out_frame, uint64_t* pts_ms);
  int md_video_decoder_width(MDVideoDecoderHandle h);
  int md_video_decoder_height(MDVideoDecoderHandle h);
  double md_video_decoder_fps(MDVideoDecoderHandle h);
  void md_video_decoder_destroy(MDVideoDecoderHandle h);
  #endif
  ```
- [ ] **Step 2**: `md_capi.cpp` 实现（`std::unique_ptr<Dag>`/`VideoDecoder` 包进句柄；`md_video_decoder_next` 内部调 `video_decoder.next(&img,&pts)` 再 `md_image_from_nv12_owned` 转 MDImageHandle）。
- [ ] **Step 3**: `tests/test_capi.cpp` 加 `[capi]` 用例（DSL 建图 execute 冒烟；视频缺文件 SKIP）。
- [ ] **Step 4**: `cmake --build build_capi` + `.\bin\test_modeldeploy.exe "[capi]"`。
- [ ] **Step 5: Commit**
  ```bash
  git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
  git commit -m "feat(capi): thin pipeline DSL + video decoder wrappers"
  ```

---

## C# / Rust（YAGNI 后续，不写全量）

- **C#** 薄封装：`ModelDeploy.Pipeline.Build(string spec)/Execute()`；`ModelDeploy.VideoDecoder.Next() -> ImageData`（DAG 内部结构不镜像；枚举/常量对齐 CAPI）。
- **Rust** 薄封装：`pipeline::Dag::build(spec)?.execute()`；`video::VideoDecoder::open()?.next()?`；`ffi.rs` extern 声明 `md_pipeline_*` / `md_video_decoder_*`（依赖 Task 8 的 CAPI；Task 8 跳过则 Rust 也顺延）。
- **demo + docs**（后续）：`examples/demo_pipeline/` 用 `det -> track -> action` 演示 DAG；`BUILD_VIDEO` 时加 `VideoDecoder` 抽帧→DAG；`EXAMPLES.md`/`README.md` 加"Pipeline DAG 编排 / 视频解码"能力。

---

## Self-Review 记录

**Spec coverage**：
- DAG Node/Edge/Dag/Planner(§3.2/3.3/3.4) → Task 1/2/3；`ModelNode<T>` 便捷子类在计划中以 `PyTransformNode`/桩节点体现，`ModelNode` 模板可后续按需（YAGNI，spec§3.2 允许只做 TransformNode）。
- VideoDecoder(§3.5) → Task 6；`BUILD_VIDEO` 默认 OFF + FFmpeg(§3.5 依赖) → Task 5。
- Python(§4) → Task 4(pipeline)/Task 7(video)。
- 测试(§9) → `[pipeline]`(Task 1-3)、`[video]`(Task 6)+ SKIP。
- CAPI(§5) → Task 8（薄，可降级）；C#/Rust(§6/7)/demo/docs(§8) → "C#/Rust/demo+docs" 节（YAGNI 后续）。交付矩阵(§10)中 C++/Python 全量，CAPI 薄(可选)，C#/Rust 标注后续。

**Placeholder 扫描**：所有代码步骤均有完整可编译代码，无 "TBD/TODO"。视频测试 SKIP 是规范允可的显式行为（无视频文件 SKIP），非占位。

**Type consistency**：
- Node 端口 `Port{name,type}`、`get_in<T>`/`set_out<T>`、`std::any` 全 Task 一致。
- `Dag::connect/ build/ execute/ execution_order`、`Planner::register_model/build`、`unique_instance` 命名在 Task 2/3/4 一致。
- `VideoDecoder::open/next(ImageData*,uint64_t*)/close/width/height/fps` 在 Task 6/7/8 全链一致；`next` 返回 `vision::ImageData` NV12 + `Device::CPU`（现状校正，无 DeviceOption）。
- CMake：`FFMPEG_INCLUDE_DIR`/`FFMPEG_LIBS` 在 Task 5/6/7 由 `cmake/ffmpeg.cmake` 产出并一致消费。
- Python：`pipeline.Dag/Planner.register_transform`（内部 `PyTransformNode`）、`video.VideoDecoder` 命名在 Task 4/7 一致，且 `video` 子模块复用 `vision.ImageData` 绑定。

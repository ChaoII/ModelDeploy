# ModelDeploy 流式 / 异步推理（AsyncModel）指南

## 为什么需要异步推理

所有模型的同步推理入口（如 `UltralyticsDet::predict`）都是**阻塞式**的：调用方必须等「预处理 → 推理 → 后处理」全部返回才能继续下一帧，无法在同一线程边投递边推进。

而视频层（`csrc/video/decode_pipeline.h`）已经具备一套成熟的异步基建：有界背压队列、双线程 `decode_loop/deliver_loop`、`FrameCallback` 回调投递。`AsyncModel` 把这套范式**下放到推理层**，让模型调用不再阻塞投递线程。

### 并发正确性铁律：同实例不可并发

`BaseModel` 内部复用成员缓冲（如 `reused_input_tensors_` / `reused_output_tensors_`），因此**同一个模型实例绝不能并发调用 `predict` / `batch_predict`**——这本身就是未定义行为（详见 [多线程推理指南](./multi_thread.md)）。

`AsyncModel` 从这里出发，做了两个关键设计决策：

1. **内部只有单一推理线程**负责调用模型方法——从根上避免同实例并发；并非对单实例上锁。
2. **吞吐靠 batch 合并 + 多实例**：把排队请求攒批后一次性 `batch_predict`；需要更高吞吐时创建多个 `AsyncModel` 对象（各自独立实例），而非在一个实例上并发上锁。

## 快速上手

```cpp
#include "pipeline/async_model.h"
#include "vision/common/image_data.h"

using modeldeploy::pipeline::AsyncModel;
using modeldeploy::pipeline::AsyncModelConfig;
using modeldeploy::vision::ImageData;

// 1. 组装配置（均为默认值示例）
AsyncModelConfig cfg;
cfg.queue_capacity = 256;               // 有界队列长度（满则投递阻塞）
cfg.num_workers    = 1;                 // 单推理线程（本期恒为 1，见下文"配置取舍"）
cfg.max_batch      = 8;                 // 单批上限
cfg.batch_timeout  = std::chrono::milliseconds(2);  // 攒批最长等待
cfg.enable_batching = true;             // 可关闭，退化为逐条 predict

// 2. 构造（模型以 unique_ptr 移交，AsyncModel 独占）
auto infer = std::make_unique<AsyncModel<MyModel>>(
    std::make_unique<MyModel>(), cfg);

// 3. 启动（启动后投递才开始被处理）
infer->start();

// 4. 投递若干帧（future 式；队列满时会阻塞等待空位，保序不丢）。image 为待推理帧
ImageData image;   // 示例占位，实际可从视频帧/读图获得
std::future<MyModel::result_type> f = infer->predict_async(image);

// 5. 等待全部完成（调试/退出前常用）
infer->wait_idle();

// 6. 停止（幂等：置停 + 唤醒 + 对队列内任务注错 + join）
infer->stop();
```

## 接口

### `AsyncModelConfig`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `queue_capacity` | 256 | 有界队列长度；满则投递阻塞（等价于解码器的 Block 策略，不丢） |
| `num_workers` | 1 | 单推理线程数；**本期恒按 1**（>1 需 `M::clone`，见下文） |
| `max_batch` | 8 | 单次 `batch_predict` 的任务上限 |
| `batch_timeout` | 2ms | 攒批时等待补齐的最长时间 |
| `enable_batching` | true | 关掉则退化为逐条 `predict` |

### `AsyncModel<M>`

模板参数 `M` 为模型类型，需满足：

```cpp
using result_type = /* 结果类型，如 std::vector<DetectionResult> */;
bool predict(const ImageData&, result_type* result, TimerArray* = nullptr);
bool batch_predict(const std::vector<ImageData>&,
                   std::vector<result_type>* results, TimerArray* = nullptr);
```

现有检测模型（如 `UltralyticsDet`）自带结果类型；接入异步层时给模型定义 `result_type` 即可。
`AsyncModel<M>::ResultType` 即 `M::result_type`——若 `M` 未定义，编译期即报错。

```cpp
template <typename M>
class AsyncModel {
public:
    using ResultType = typename M::result_type;
    using Callback = std::function<void(uint64_t task_id,
                                        ResultType&& result,
                                        const std::string& error)>;

    AsyncModel(std::unique_ptr<M> model, const AsyncModelConfig& cfg = {});

    // 生命周期（start 后投递才受理；stop 幂等，可任意线程调用）
    bool start(std::string* err = nullptr);
    void stop();

    // future 式：队列满阻塞等待空位（保序不丢），返回 future 供取结果
    std::future<ResultType> predict_async(const ImageData& image,
                                          std::string* err = nullptr);

    // 回调式：同队列；结果在推理线程回调（携带 task_id）
    void set_callback(Callback cb);                                   // start 前设置，可重复替换
    uint64_t predict_async_cb(const ImageData& image, std::string* err = nullptr);  // 返回 task_id

    // 可观测
    bool busy() const;          // == pending() > 0
    uint64_t pending() const;   // 队列中待处理数
    uint64_t submitted() const; // 累计提交数
    uint64_t completed() const; // 累计完成数（含失败）
    uint64_t batch_runs() const;// 累计执行 batch_predict 次数

    void wait_idle();           // 阻塞到全部完成（测试/调试用）
};
```

### `AsyncVideoInfer<M>`（解码 → 异步推理便捷接线）

```cpp
namespace modeldeploy::video;

template <typename M>
class AsyncVideoInfer {
public:
    explicit AsyncVideoInfer(modeldeploy::pipeline::AsyncModel<M>& infer);

    void on_frame(VideoFrame&& frame);          // 供解码器接线：零拷贝转发 frame.image
    void set_result_callback(Callback cb);      // 转投 AsyncModel::set_callback
    uint64_t frames_submitted() const;          // 已投递帧数
};
```

`AsyncVideoInfer` 把解码器逐帧回调（`FrameCallback: void(VideoFrame&&)`）直接接线到异步推理：每帧取出内嵌的 `frame.image`（零拷贝）投递到 `AsyncModel`，不做任何转换。**它不拥有** `AsyncModel`，生命周期由调用方管理（先 `AsyncModel::stop()` 再析构）。

## 两种结果通道

`AsyncModel` 同时提供 **future 式**与**回调式**两条通道，可在同一实例上混用：

| 通道 | 入口 | 结果获取 | 适用 |
|------|------|---------|------|
| future | `predict_async` | 任意线程 `future.get()`（同步等待） | 简单、一次等一个结果 |
| 回调 | `predict_async_cb` + `set_callback` | 推理线程异步回调，携带 `task_id` 与 `error` | 高吞吐、不阻塞投递 |

回调签名：`void(uint64_t task_id, ResultType&& result, const std::string& error)`。

- **保序**：`num_workers == 1`，回调按提交顺序返回。
- **错误处理**：单任务 `predict` / `batch_predict` 返回 `false` 或抛异常 → future `set_exception`，回调带 `error` 字符串；**其余任务不受影响**。
- **线程安全**：`predict_async` / `predict_async_cb` 可被任意多线程并发调用；`stop()` 幂等、可在任意线程调用。

## 配置取舍说明（路线图）

`num_workers` 字段为扩展预留，但**本期恒按 1**。原因：同一模型实例不可并发，`num_workers > 1` 需要 `M::clone()` 克隆多个实例分摊任务；`M::clone()` **本期未实现**，故多 worker 属于路线图内容。

当前要提升吞吐，正确做法是：

- 靠 `max_batch` / `batch_timeout` 的**攒批**吃满单实例 batch 能力；
- 需要更高并行度时创建**多个 `AsyncModel` 实例**（各自独立模型实例），互不干扰。

## 限制（本期范围）

本期 `AsyncModel` 能力与边界：

- **仅 C++**：Python / C / C# / Rust 等其它语言绑定后续再议。
- **同实例并发缺失**：单实例内部仍是单推理线程；并行靠多实例。缓冲池化（buffer pooling）列入远期。
- **跨帧有状态算子（stateful）后续期**：本期待处理的帧彼此独立，不保留跨帧状态。
- **背压策略仅 Block**：队列满时投递阻塞保序，不提供 Drop / Overwrite 切换（如需要后续加枚举）。

## 相关文档

- [多线程推理指南](./multi_thread.md) — 同实例不可并发的底层原理与 `clone()` 方案
- [架构设计](./architecture.md) — `BaseModel` 核心抽象与推理链路
- [视频编解码总览](./video/README.md) — 解码器背压与回调基建

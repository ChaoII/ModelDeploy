# ModelDeploy 多线程推理指南

## 模型实例是线程不安全的

同一个 `Model` 对象（如 `UltralyticsDet`、`Scrfd` 等）**不能**在多线程中同时调用 `predict()`。

```cpp
// ❌ 错误：同一实例多线程并发 → 数据竞争 → 崩溃
auto& model = get_model();
thread1: model.predict(img1, &r1);  // 写 binding_
thread2: model.predict(img2, &r2);  // 同时写同一 binding_ → 竞争
```

根本原因：每个 `OrtBackend` 实例只有一个 `IoBinding`，`predict()` 过程中会反复 `BindInput→Run→GetOutput`，多线程同时操作同一个 `binding_` 是未定义行为。

## 正确的并发方式：clone

用 `model.clone()` 给每个线程创建独立实例。克隆体共享 ORT Session（轻量），各自持有独立的 `IoBinding`（线程安全）。

```cpp
// ✅ 正确：每个线程 clone 一份
auto c1 = model.clone();
auto c2 = model.clone();
thread1: c1->predict(img1, &r1);  // binding_A，安全
thread2: c2->predict(img2, &r2);  // binding_B，安全（同时执行）
```

### clone 做了什么

```
原始 OrtBackend
  ├─ shared_session_ (shared_ptr<Ort::Session>)  ← 克隆体共享
  └─ binding_         (unique_ptr<IoBinding>)     ← 独占

克隆体 OrtBackend
  ├─ shared_session_ ───指向同一个 Session───      ← 轻量共享
  └─ binding_         (NEW IoBinding)              ← 各自独占
```

`clone()` 开销很小：只新建一个 `IoBinding`、拷贝输入输出描述信息。不重新加载模型、不重新编译 TRT engine。

### 完整示例

```cpp
// 1. 加载模型（一次）
modeldeploy::RuntimeOption opt;
opt.use_gpu(0);
UltralyticsDet model("yolo.onnx", opt);

// 2. 对每个线程 clone 一份
std::vector<std::unique_ptr<UltralyticsDet>> clones;
for (int t = 0; t < THREAD_COUNT; ++t)
    clones.push_back(model.clone());

// 3. 多线程推理
std::vector<std::thread> threads;
for (int t = 0; t < THREAD_COUNT; ++t) {
    threads.emplace_back([&clones, t, &img]() {
        std::vector<DetectionResult> r;
        clones[t]->predict(img, &r);
    });
}
for (auto& th : threads) th.join();
```

### 使用 `clone()` 注意事项

| 项目 | 说明 |
|------|------|
| 调用 `clone()` 的时机 | 在**启动线程之前**完成所有 clone，不要在运行时反复 clone |
| clone 数量 | 通常 = 线程数，每线程持有自己的克隆体 |
| 克隆体寿命 | 与线程同生命周期，线程结束后析构 |
| 线程安全 | 不同克隆体的 `predict()` 可完全并行 |
| 预处理器设置 | `clone()` 会复制 pre/post processor，如需自定义参数在 clone **之后**单独设置 |

### 什么时候不需要 clone

| 场景 | 方案 |
|------|------|
| 单线程推理 | 直接用原始模型，不需要 clone |
| 多路相同模型 | 每个 Pipeline 通过 `PipelineManager::create_engine()` 自动 clone |
| 不同模型 | 本来就是不同的实例，各自独立 |

### 性能对比

| 方案 | FPS（4 线程，GPU） | 说明 |
|------|-------------------|------|
| 同一实例并发 | 崩溃（UB） | binding_ 数据竞争 |
| clone 后并发 | **≈ 4× 单线程** | 共享 Session，独占 binding，GPU 满吞吐 |

真实性能与模型大小、GPU 型号、batch size 相关。4 × YOLO11n 在 RTX 4060 Ti 上实测约 **650 FPS**。

## `clone()` 的实现架构

```
UltralyticsDet::clone()
  └─ clone_runtime() → Runtime::clone()
       └─ OrtBackend::clone()
            ├─ shared_session_ = this->shared_session_   (共享)
            ├─ model_buffer_    = this->model_buffer_     (拷贝)
            ├─ inputs_desc_     = this->inputs_desc_      (拷贝)
            ├─ outputs_desc_    = this->outputs_desc_     (拷贝)
            └─ binding_ = new IoBinding(*shared_session_) (新建)
```

`clone()` 不涉及任何锁或同步原语。线程安全完全依靠「每个线程持有自己的克隆体」这个架构设计。

查看更多代码示例：
- `examples/demo_det/demo_detection_cxx.cpp` — 单线程基础用法
- `examples/demo_det/demo_multi_thread_compare.cpp` — 多线程对比演示

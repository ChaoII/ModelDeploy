# capiv2 与 C#/Rust 绑定对比分析

> 主题：`capi`（C API v2）+ 它的 C#（`csharp/ModelDeploy/V2`）与 Rust（`rust/modeldeploy`）
> 绑定，对比原始的 `capi/`（C API v1）绑定方案；
> 并横向对比另一个分支 `tensor-decouple-device-memory` 的核心重构。
>
> 面向读者：SDK 维护者、绑定层开发者、对跨语言 ABI 设计感兴趣的人。
> 结论先行：**capiv2 不是"表面改改"的接口重命名，而是一次面向 ABI 稳定、多语言零冗余、
> 内存所有权统一、设备侧零拷贝的"契约级"重构**。
>
> 配套可落地的操作指南见 [capi_cpp_language_binding_best_practices.md](capi_cpp_language_binding_best_practices.md)
> （C 封装 C++、供多语言绑定的最佳实践，含参数设置/前后处理参数/踩坑清单/范例）；
> 已知设计缺陷与风险登记见 [capi_risk_register.md](capi_risk_register.md)。

---

## 1. 一句话结论

| 对比项 | 原始 capi（v1） | capiv2 + 新绑定 |
|--------|----------------|-----------------|
| 本质 | 每个模型一把独立的裸 C 函数族，结构体对外暴露 | 一套"不透明句柄 + 单一分发 + 统一所有权 + blittable 结果"的稳定 C ABI |
| 代码规模 | 61 个文件 / 约 4581 行，散布于 `capi/**` | 2 个主文件 / 约 2124 行（`capi/md_capi.{h,cpp}`） |
| 绑定成本 | 每个模型都要在 C#/Rust 里写一套样板 | 每个语言只需"1 个通用基类 + 若干结果读取器"即可全覆盖 |

**从软件工程的角度讲，capiv2 干的这件事，在现代对应几个很明确的说法：**
以稳定 C ABI 为边界的**多语言绑定（FFI bindings）**、**门面（Facade）模式**、
**不透明句柄 / PIMPL 惯用法**、**防腐层（Anti-Corruption Layer）**、
配合 **RAII** 与 Rust 的**"安全 API 包裹不安全 FFI"**惯用法。
而 `tensor-decouple-device-memory` 分支则是另一个**正交维度**的内部重构（详见第 9 节），
两者恰好互补。

---

## 2. 原始 capi（v1）设计复盘：它"闹心"在哪

原始 `capi/` 是按**模型类型**组织的一批 C 接口，例如：

```c
// capi/vision/detection/detection_capi.h
MDStatusCode md_create_detection_model(MDModel* model, const char* path, const MDRuntimeOption* opt);
MDStatusCode md_detection_predict(const MDModel* model, MDImage* image, MDDetectionResults* res);
void md_free_detection_result(MDDetectionResults* c_results);
void md_free_detection_model(MDModel* model);
```

### 2.1 结构性痛点

1. **结构体对外暴露，可被调用方随便改**
   `MDModel` 里直接有 `type / format / model_content(char*) / model_name` 等字段，
   `MDDetectionResults` 也直接有 `size / data` 指针。跨语言调用方可以随意
   强转、改字段，一旦读写错位就是 `undefined behavior`（未定义行为），
   而且是那种"最隐蔽、最难查"的内存型问题。

2. **每个模型都要写/抄一遍函数族样板**
   检测、分类、姿态、OBB、OCR、LPR、人脸……每个都重复
   `md_create_xxx / md_set_xxx_input_size / md_xxx_predict / md_xxx_predict_nv12 /
   md_print_xxx / md_draw_xxx / md_free_xxx` 一整组。表现在代码上就是
   61 个文件、约 4581 行，绝大部分是复制粘贴出来的"重复样板"。

3. **内存所有权混乱、极易配错**
   同一个库里混合出现 `malloc / new[] / strdup / delete[] / free`：
   ```cpp
   model->model_name = strdup(...);   // strdup -> 该用 free
   results->data = new X[n];          // new[]  -> 该用 delete[]
   c_results->data = delete[]();       // ...
   ```
   调用方被迫记住"谁负责释放、该用哪个"，漏配一个就内存泄漏/崩溃。

4. **没有统一的错误模型**
   每个函数返回自己的状态码，没有"最近一次错误信息"的统一通道，
   排查问题时只能靠日志，跨语言侧拿不到可供展示的失败原因。

5. **结果传递靠"手动结构体 + 手动 marshal"**
   C# 侧被迫 `Marshal.AllocHGlobal` + `StructureToPtr` 把托管结果**塞回**原生结构体，
   或用 `GCHandle.Alloc(..., Pinned)` 手动钉住 NV12 平面再传指针，
   `finally` 里再一层层 `Free`。绑定层非常脆、样板极重、极易泄漏。

### 2.2 绑定层被"传染"的表现

原始 C# 绑定（`main` 分支 `csharp/ModelDeploy/vision/**`）直接镜像上面的每模型函数：

```csharp
// main:csharp/ModelDeploy/vision/det/models.cs
_model = new MDModel();
Utils.Check(md_create_detection_model(ref _model, modelDir, ref nativeRuntimeOption), "...");
// 手动维护 MDDetectionResults，最后手动 md_free_detection_result(ref cResults)
```

于是**原生 C 层每多一个模型，C# 就得跟着写一个 class + 一组结构体 + 一组 marshal 样板**，
Rust 同样。这是典型的"重复样板随模型数量线性增长"，维护成本爆炸。

---

## 3. capiv2：革命性的契约重构

`capi/md_capi.h` 的注释直接点明了设计原则（这就是"革命性"的官方措辞）：

> 1. 不透明句柄：调用方永远接触不到库内部指针/结构体字段，杜绝类型强转与字段篡改。
> 2. 单一分发点：模型创建/释放/推理各自只有一个入口，内部按类型分发，消灭重复样板。
> 3. 统一内存所有权：句柄一律由库分配/释放，结果统一 `md_result_destroy`。
> 4. 纯 C99：头文件不依赖 C++ 语法，任何 C 编译器可编译。
> 5. 富错误模型：`MDStatus` 覆盖参数/状态/并发类错误，`md_get_last_error()` 线程安全。
> 6. 线程模型：每个句柄单线程，库内不加锁、零开销。
> 7. 数组式结果访问：固定字段结果一次取回 blittable 结构体数组（零拷贝）。

### 3.1 五大核心武器

**(1) 不透明句柄（Opaque Handle）**

```c
typedef struct md_model_handle*   MDModelHandle;
typedef struct md_image_handle*   MDImageHandle;
typedef struct md_result_handle*  MDResultHandle;
typedef struct md_option_handle*  MDOptionHandle;
```

调用方只见指针、看不到内部。C# 用 `IntPtr`，Rust 用 `NonNull<c_void>` 包装。
杜绝了 v1 那种"结构体裸奔 + 任意改写"的灾难。这正是经典 **PIMPL / 句柄（handle）** 惯用法。

**(2) 单一分发点（Single Dispatch / Facade）**

25+ 种模型，创建只有一个入口，内部按 `MDModelKind` 分发：

```c
MDStatus md_model_create(MDModelHandle* out, MDModelKind kind,
                         const char* path, const MDOptionHandle opt);
MDStatus md_model_predict(MDModelHandle, MDImageHandle, MDResultHandle* out);
void      md_model_destroy(MDModelHandle);
```

多子模型（OCR / LPR pipeline / insightface / ASR / TTS）用 `'|'` 分隔符串联在一个字符串里。
**新增一种模型 = 加一个枚举 + 在分发里加一个 case**，绑定层完全不用改。
这就是"消灭重复样板"的根因。

**(3) 统一内存所有权**

- 图像：`md_image_from_*` 由库分配，统一 `md_image_destroy`；
- 结果：`md_model_predict` 由库分配，统一 `md_result_destroy`；
- 输入（`from_bgr24/rgb24/nv12/from_encoded` 等）由调用方持有、库只引用不拷贝。

调用方永远只需要配对"一个 create / 一个 destroy"，不用再判断 `malloc/free/new[]/delete[]/strdup`。

**(4) Blittable 结果结构 + 数组式 getter（零拷贝）**

固定字段结果（框、分数、类别）被设计成**纯数值、无指针**的结构体，一次取回整个数组：

```c
typedef struct MDDetectionItem { float x,y,w,h; float score; int label_id; } MDDetectionItem;
MDStatus md_result_detection(MDResultHandle, const MDDetectionItem** items, size_t* count);
```

这套结构体是 **blittable**（内存布局可直接 1:1 映射，C#/Rust 无需序列化）——
这正是第 4、5 节绑定零拷贝 marshal 的关键前提。

**(5) 富错误模型 + 线程安全错误通道**

```c
typedef enum MD_STATUS { MD_OK, MD_ERR_NULL_POINTER, ..., MD_ERR_BUSY, ... } MDStatus;
const char* md_get_last_error(void);   // thread_local，线程安全
```

还把"句柄被并发使用"这种潜在的 UB 也显式建模（`MD_ERR_BUSY`），
明确"每个句柄单线程、多线程各自 create"的线程模型，并把"不加锁"作为性能承诺写进设计。

### 3.2 代价 / 反手

- **接口从"直白"变成"间接"**：调用方要理解 `MDModelKind`、`MDResultKind`、`'|'` 分隔路径约定。
- **灵活性收窄**：v1 那种"拿到底层指针随便折腾"的高级玩法没了，这是有意为之（换安全）。
- **一次"从 0 到 1"的重写成本**：所有既有 C/C++/C#/Rust 用户都要迁移。

---

## 4. C# 绑定：从"手动 marshal 全家桶"到"门面式强类型封装"

### 4.1 新绑定结构（`csharp/ModelDeploy/V2/`）

- `NativeMethods.cs`：**唯一**的 `DllImport` 集中地，全部函数声明收敛到一个 `partial class`。
  所有句柄一律 `IntPtr`，所有结果项结构体 `blittable`。
- `types_internal_c.cs`：与 `md_capi.h` 对齐的枚举 + blittable 结构体。
- `BaseModel.cs`：
  - `RuntimeOption2`：**链式 setter（fluent）** 的运行时选项，替代 v1 的 `MDRuntimeOption` 结构体直译；
  - `BaseModel : IDisposable`：抽象基类，统一 `md_model_create / IsReady / Dispose`；
  - `ResultReader`：封装"取数组指针 → 逐个 `Marshal.PtrToStructure<T>` → 统一释放"，
    一个泛型方法搞定所有模型的结果读取。
- `Prediction<T>.cs`：`IReadOnlyList<T> + IDisposable`，**惰性缓存**结果列表，
  同时保留 `Handle` 供 `Draw()` 句柄直达 C++ `vis_*`。
- `Results.cs`、`Models.cs`：强类型结果对象与各模型派生类。

### 4.2 关键样板对比

**v1（每个模型都要）**：
```csharp
private MDModel _model;
var cResults = new MDDetectionResults();
Utils.Check(md_detection_predict(ref _model, ref image.RawImage, ref cResults), "...");
try { return new List<DetectionResult>(DetectionResult.FromNativeArray(cResults)); }
finally { md_free_detection_result(ref cResults); }
```
还要手动 `MDDetectionResult.ToNativeArray`、`GCHandle.Alloc(...Pinned)` 等。

**v2（一次写成，通用）**：
```csharp
// 核心只写一遍
protected Prediction<T> MakePrediction<T>(VisionImage image, Func<IntPtr,T[]> reader)
    => new Prediction<T>(PredictNative(image.Handle), reader);
// ResultReader.ReadItems<T> 一个方法循环读任意 blittable 数组
```
新增一个模型 = 新增一个继承 `BaseModel` 的类 + 一个结果读取 lambda，**不再有各家手工 marshal**。

### 4.3 收益

- `IDisposable + ~析构` + `GC.SuppressFinalize` 让原生句柄生命周期跟托管对象绑定（RAII 化）。
- 句柄直达 `md_draw_result`，可视化"零重建、最忠实"。
- `Prediction<T>` 惰性读取：不需要可视化调用方时零开销。

---

## 5. Rust 绑定：用语言特性把 FFI 做成"安全且优雅"

Rust 的 `rust/modeldeploy/src/` 是这套哲学的**最佳体现**：

- `ffi.rs`：与 `md_capi.h` 1:1 的 `extern "C"` 声明。
- `model.rs`：核心 `Model` 结构 + `model_wrapper!` 宏，**用宏一次性生成全部 20+ 个模型包装**：
  ```rust
  macro_rules! model_wrapper {
      ($name:ident, $kind:expr, $reader:expr) => {
          pub struct $name { inner: Model }
          impl $name {
              pub fn new(model_path, option) -> Result<Self, MdError> { ... }
              pub fn predict(&self, image) -> Result<Vec<<$name as ResultType>::Item>, MdError> { ... }
              // ...
          }
      };
  }
  model_wrapper!(UltralyticsDet,  ModelKind::Detection, RawResult::detection);
  model_wrapper!(PaddleOCR,        ModelKind::Ocr,       RawResult::ocr);
  // ... 一行一个模型
  ```
  - **泛型 + trait（`ResultType`）**决定每个模型的返回类型，静态类型安全；
  - 深拷贝（`Model::clone`）复用已加载的后端 session。
- **RAII**：`impl Drop for Model` / `impl Drop for RawResult` 里 `md_model_destroy / md_result_destroy`，
  句柄随所有权自动释放，不可能泄漏。
- **安全 API 包裹不安全 FFI**：`unsafe { ffi::md_* }` 全部收敛在内部，对外只暴露 `Result<_, MdError>`；
  `unsafe impl Send/Sync` 显式声明这些句柄可并发传递（配合 capiv2 的"每句柄单线程"契约）。
- `RawResult` 透出 `capi` 的每种结果读取方法（detection / pose / ocr / lpr / embed 等）
  并自动做 `*const` → `Vec<T>` 的零拷贝切边。

这套写法在 Rust 生态里就是标准的 **"安全 wrapper 包裹 unsafe FFI + RAII 管理资源"**
（navigation：即 FFI 绑定的最佳实践，`raw pointer` 只在边界出现一次）。

---

## 6. result 结构体的使用 / 传递 / 解析（数据流详解）

这里深入 capiv2 的结果句柄（`MDResultHandle`）——它是一切"零拷贝、统一所有权"
承诺真正落地的地方。核心实现见 `capi/md_capi.cpp` 约 1293–1699 行。

### 6.1 三段式生命周期

```
md_model_predict(...)   ──►  MDResultHandle（库内分配）
        │                        ▲
        │      md_result_count / 固定字段 getter / 变长项 getter（按索引取原始指针）
        └────────────────────────┘
md_result_destroy(h)    ──►  一次性释放（无论调过几个 getter，都不漏）
```

使用者始终不碰 C++ 类型，只拿"句柄 + 纯数值 blittable 结构体数组 + 内部指针"。

### 6.2 内部怎么"存"：一套泛型容器

结果句柄本体（`md_capi.cpp:79`）：

```cpp
struct md_result_handle {
    MDResultKind kind;   // 标记结果种类
    void* data;          // ResultDataBase*（具体类型由 kind 决定）
};
```

统一基类 + 两种载体：

- **列表结果**：`ResultData<T>{ std::vector<T> v }`，`T` 直接是内核 C++ 类型
  （`DetectionResult / KeyPointsResult / LprResult / ...`）；
- **单值结果**：`SingleResult<T>{ T value }`（sem_seg / depth / age / gender / OCR）。

Predict 时（`md_capi.cpp:820`）核心只有一句，**原生结果零深拷贝地落进 vector**：

```cpp
auto* d = new ResultData<DetectionResult>();
m->predict(image, &d->v);      // 复用 C++ 内核 predict
rh->kind = MD_RES_DETECTION;
rh->data = d;
```

#### 6.2.1 深度讲解：`ResultData` 与 `ProjectedResult` 两层容器（完整源码）

这两个是结果句柄内部最核心、也最容易绕晕的部分。先看完整原始定义（`md_capi.cpp:85–143`）：

```cpp
struct ResultDataBase {
    virtual ~ResultDataBase() = default;            // 多态：让 void* data 能统一 delete
    virtual size_t count() const = 0;               // 统一取个数
};

// 列表结果：直接持有内核 C++ 结果
template <typename T>
struct ResultData : ResultDataBase {
    std::vector<T> v;
    size_t count() const override { return v.size(); }
};

// 单值结果（sem_seg / depth / age / gender / OCR 整页）
template <typename T>
struct SingleResult : ResultDataBase {
    T value{};
    size_t count() const override { return 1; }
};

// —— 下面是"给 C 看的投影" ——
struct ProjectedResultBase : ResultDataBase {
    virtual ResultDataBase* origin_ptr() const = 0; // 暴露 origin（供 raw_result 取回）
};

template <typename Dst>
struct ProjectedResult : ProjectedResultBase {
    std::vector<Dst> v;          // 投影后的 blittable 数组（给 C 读）
    ResultDataBase* origin = nullptr;  // 回指原始 ResultData<T>*（所有权：析构时 delete）
    size_t count() const override { return v.size(); }
    ResultDataBase* origin_ptr() const override { return origin; }
    ~ProjectedResult() override { delete origin; }   // ← 所有权链关键
};

// 在"原始 ResultData / 已投影的 origin"里，按类型找原始仓库
template <typename T>
ResultData<T>* raw_result(md_result_handle* rh) {
    auto* base = static_cast<ResultDataBase*>(rh->data);
    if (auto* d = dynamic_cast<ResultData<T>*>(base)) return d;
    if (auto* p = dynamic_cast<ProjectedResultBase*>(base)) {
        return dynamic_cast<ResultData<T>*>(p->origin_ptr());
    }
    return nullptr;
}
```

**每个容器的职责**：

| 容器 | 存什么 | 何时生成 | 一句话职责 |
|------|--------|----------|-----------|
| `ResultDataBase` | — | — | 多态基类：让 `void* data` 能统一 `delete`、统一 `count()` |
| `ResultData<T>` | 内核 C++ 结果 `std::vector<T>` | predict 阶段 | **原始数据仓库**（如 `vector<DetectionResult>`） |
| `SingleResult<T>` | 单个 `T value` | predict 阶段 | 单值结果的特殊仓库（不含 vector） |
| `ProjectedResultBase` | — | — | 投影容器的多态基类，暴露 `origin` |
| `ProjectedResult<Dst>` | blittable 数组 `std::vector<Dst>` + `origin` | 第一次 getter | **投影/缓存/回链三合一** |
| `raw_result<T>()` | — | getter 内部 | 按类型找回原始仓库 |

**为什么一定要"两层"？** 因为同一个结果句柄要同时满足三个互相拉扯的需求：

1. **给 C 一个能读的形态** → C 不能直接消费 C++ 的 `DetectionResult`（它可能含 `Rect2f`、
   甚至 `std::vector<std::string>` 等 C++ 特有类型），必须投影成 blittable 的 `MDDetectionItem`；
2. **要零拷贝** → 投影"只算一次"，缓存进 `p->v`，之后每次 getter 直接给指针，不重算；
3. **要留原始数据给变长字段** → 关键点 / mask / 字符串**不在** blittable 数组里，它们在
   `ResultData<KeyPointsResult>::v[i].keypoints` 里。一旦投影就把原始仓库扔掉，变长 getter 就没数据可给。

`ProjectedResult` 一箭三雕：**数组投影（给 C）+ 缓存（零拷贝）+ `origin` 回链（留原始）**。
对应地，变长 getter（如 `md_result_mask`）就靠 `p->origin` 找回原始仓库、拿内部指针：

```cpp
auto* origin = static_cast<ResultData<InstanceSegResult>*>(p->origin);
*buf = origin->v[i].mask.buffer.data();   // 零拷贝给库内指针
```

**完整时序（一次 predict → getter → destroy）**：

```
predict:
  rh->data = ResultData<DetectionResult>{ v:[det0, det1, ...] }   ← 原始仓库

第一次 getter(md_result_detection):
  rh->data = ProjectedResult<MDDetectionItem>{ v:[...], origin: ResultData... }
             ├─ *items 给 MDDetectionItem 数组（零拷贝借用）    ← 需求1+2
             └─ origin ──→ 变长 getter 从这里拿真实 C++ 字段指针  ← 需求3

md_result_destroy:
  delete ProjectedResult → 析构里 delete origin(ResultData)
  → 一条链全部释放，不泄漏
```

**所有权链（物理版"一次 destroy 不漏"）**：`ProjectedResult` **拥有** `origin`。所以即使
`rh->data` 从 `ResultData` 换成了 `ProjectedResult`，最终只 `delete` 一次 `ProjectedResult`，
它再 `delete` 掉 `origin`。无论你调没调 getter、调了几个变长 getter，链只有一条。

**澄清 getter 首行那个 `static_cast`**：
```cpp
md_result_detection(...) {
    auto* d = static_cast<ResultData<DetectionResult>*>(rh->data);  // 假定还是"原始仓库"
    ...
}
```
它成立的**前提**是"这次 getter 是第一次调用，`rh->data` 尚未被投影成 `ProjectedResult`"。
这是数组 getter（detection/classification/obb/... 等）当前采用的**简单写法**，代价是：
**同一结果句柄把同一数组 getter 调第二次会 static_cast 类型不符、读错内存**（见 6.6 可改进点）。
而 `raw_result<T>()` 的存在就是为了兼容"已投影 / 未投影"两种状态——`md_result_face`、
`md_result_lpr` 用的正是它，因此能一个 getter 兼容多种底层 C++ 类型（如 `md_result_face`
既读 `KeyPointsResult`（Scrfd）也读 `InsightFaceBox`（insightface-det））。

### 6.3 怎么"传"：投影 + 变长项双通道

**A. 固定字段 → 投影成 blittable 数组，一次给指针（零拷贝）**

以检测为例（`md_capi.cpp:1295`）：

```cpp
MDStatus md_result_detection(MDResultHandle h, const MDDetectionItem** items, size_t* count) {
    auto* d = static_cast<ResultData<DetectionResult>*>(rh->data);
    auto* p = new ProjectedResult<MDDetectionItem>();      // 投影容器
    for (const auto& r : d->v) {
        MDDetectionItem it{};
        it.x=r.box.x; it.y=r.box.y; it.w=r.box.width; it.h=r.box.height;
        it.score=r.score; it.label_id=r.label_id;          // C++→纯数值，无指针
        p->v.push_back(it);
    }
    p->origin = d;                  // 记住原始 C++ 容器（所有权转移，见 6.5）
    rh->data = p;                   // 缓存投影结果
    *items = p->v.data();           // 直接给库内 vector 内部指针
    *count = p->v.size();
    return MD_OK;
}
```

`MDDetectionItem` 是纯数值 blittable 结构体（float/int，无指针），跨 C#/Rust **1:1 内存映射，
零序列化**；`*items = p->v.data()` 返回库内 vector 的首元素指针，调用方只是"借用"读取。

**B. 变长数据 → 留在原始容器，按索引回传指针（零拷贝）**

因为关键点/掩码/embedding/字符串无法平铺进 blittable 结构体，它们**不复制**、始终待在
原始 `std::vector` 里，getter 通过 `ProjectedResult::origin` 找回原始容器再给指针：

```cpp
md_result_keypoints(...)          // 姿态骨架 → origin->v[i].keypoints.data()
md_result_mask(...)               // 实例分割掩码 → origin->v[i].mask.buffer.data()
md_result_plate(...)              // 车牌字符串 → origin->v[i].car_plate_str.c_str()
md_result_face_embedding(...)     // 人脸编码 → origin->v[i].embedding.data()
```

这跟 v1 手写 `new[] + strdup` 深拷贝成 C 结构截然不同。`raw_result<T>()` 帮助函数
（`md_capi.cpp:135`）用 `dynamic_cast` 在"原始 ResultData / ProjectedResult 的 origin"之间
按类型批配，让**同一个 getter 兼容多种底层 C++ 类型**（如 `md_result_face` 既能读
`KeyPointsResult` 也能读 `InsightFaceBox`；`md_result_lpr` 既能读 `LprResult` 也能读
`KeyPointsResult`）。

### 6.4 怎么在 C# / Rust 里"解析"

**C#（`BaseModel.cs` 的 `ResultReader.ReadItems<T>`）**：拿到 items 指针 + count 后，
blittable 结构体布局一致，直接 `Marshal.PtrToStructure<T>(IntPtr.Add(items, i*stride))`
逐项拷成托管数组；变长数据再调 B 类 getter 用 `Marshal.Copy` 拷成 `float[]/byte[]`。

**Rust（`model.rs` 的 `RawResult::detection`）**：拿到指针后
`std::slice::from_raw_parts(items, n)` 切成切片**零拷贝读**，unsafe 只在这一层出现一次，
再 `map` 成强类型 `Detection { rect, label_id, score }`。

### 6.5 所有权与释放：为什么"一次 destroy 保证不漏"

释放链：

```
md_result_destroy(h)  →  delete md_result_handle  →  delete ResultDataBase*
                                                          │
                        若 data 是 ProjectedResult：析构里 delete origin（原始 ResultData<T>）
```

所以无论调用方：调过数组 getter（data 变成 `ProjectedResult`）、一个 getter 都没调
（仍是原始 `ResultData<T>`）、只读了变长项（origin 始终被 `ProjectedResult` 持有）——
**所有权链只有一条、只 `delete` 一次，绝不泄漏**。这正是 v1 那套"`delete[] data` 却漏掉
strdup/new"的痛点被从机制上根除的地方。

### 6.6 实现层的一个可改进点（不影响设计成立）

数组 getter 以 `static_cast<ResultData<T>*>(rh->data)` 开头，**假设 data 仍是原始
ResultData**。若对同一结果句柄**重复调用同一个数组 getter 两次**，第二次时 `rh->data`
已是 `ProjectedResult`，该强转类型不符会读错内存。实际 C#/Rust 中每个结果句柄的每个
getter 通常只调一次（读完后句柄即 `destroy`），故此不会触发。若想更稳，可让数组 getter
也走 `raw_result<T>(rh)` 判类型（与 face/lpr 一致），重复调用即安全——一个低成本的小改进。

---

## 7. 优劣总表：capiv2 + 新绑定 vs 原始 capi

| 维度 | 原始 capi（v1） | capiv2 + C#/Rust 新绑定 |
|------|----------------|--------------------------|
| ABI 契约 | 结构体暴露、可被改 | 不透明句柄，外部不可达内部 |
| 接口面 | 每模型一组函数族（61 文件） | 单一分发点（2 文件） |
| 新增模型成本 | 原生+每种绑定各写一套样板 | 原生加 1 枚举 case；绑定层几乎不用动 |
| 内存所有权 | malloc/new/strdup/free 混用 | 统一 create/destroy，库内自洽 |
| 错误 | 状态码散落，无统一错误通道 | 富 `MDStatus` + 线程安全 `md_get_last_error` |
| 结果传递 | 手动结构体 + 手动 marshal | blittable 数组 + 零拷贝 getter |
| C# 绑定 | 每模型手工 GCHandle/marshal | BaseModel + ResultReader 泛型一次搞定 |
| Rust 绑定 | 每模型重复 unsafe 样板 | macro + trait 一次生成，RAII Drop |
| 线程模型 | 隐含、不安全 | 显式"每句柄单线程 + MD_ERR_BUSY" |
| 灵活性 | 高（可乱动底层） | 受控（换取安全与稳定） |
| 迁移成本 | 基线 | 一次性重写，用户需迁移 |
| 可读样本 | 4581 行样板 | 2124 行统一契约 |

**革命性的本质**：把"接口数量随模型数量线性增长"重构成"接口数量固定为常量，
模型只是数据（一个枚举）"。这让**C/C#/Rust 三种语言永远只需要维护一套通用外壳**。

---

## 8. 这在最新软件工程中叫什么？

一句话：**"以稳定 C ABI 为边界的多语言绑定（multilanguage binding over a stable C ABI）"**，
它由以下一组公认的模式/惯用法组成：

1. **Facade（门面模式）** —— 单一分发点 `md_model_create/predict/destroy` 是典型门面，
   把 25+ 种模型背后的复杂性收敛到最少入口。
2. **Opaque Handle / PIMPL（不透明句柄 / 指正实现）** —— `struct md_model_handle*` 只声明不定义，
   编译期隔离实现，是 C ABI 里"信息隐藏 + 二进制稳定性"的标准手法。
3. **Anti-Corruption Layer / 防腐层（DDD）** —— 稳定的 C API 作为内核与外部语言的"隔离带"，
   将内核 C++ 对象模型翻译成跨语言安全的 C 契约，防止内核演进"污染"各语言绑定。
4. **Stable ABI（稳定二进制接口）** —— 纯 C99、固定导出符号、固定结构体布局，保证跨编译器/跨语言兼容。
5. **RAII 资源管理** —— 库内"统一所有权"；C# 用 `IDisposable`/析构、Rust 用 `Drop` 承接。
6. **Safe wrapper over unsafe FFI（安全外壳包裹不安全 FFI）** —— Rust 特有：
   把 `unsafe` 收敛到边界，对外只露 `Result<_, MdError>`；`unsafe impl Send/Sync` + 单线程句柄契约。
7. **泛型 / 接口抽象（源码级去重复）** —— Rust 的 `model_wrapper!` 宏 + `ResultType` trait、
   C# 的 `BaseModel` + `Prediction<T>`，本质是"把模型类型做成类型参数"，
   消灭每模型重复样板（DRY）。
8. **零拷贝（Zero-copy）** —— blittable 结构体数组 + 指针透传（尤其 NV12 设备帧），
   跨语言往返零序列化开销。

> 补充一句接地气的比喻：capiv2 就好比把"每种产品都单独开一条全自动装配线"
> 改成"一条通用装配线 + 每种产品只是一份配方（枚举）"；
> 用一句行业黑话讲，这就是 **"面向 ABI 的泛型化重构（ABI-generic refactoring）"**。

---

## 9. 对照：`tensor-decouple-device-memory` 分支在做什么

**关键提醒：这是与 capiv2 完全不同的"正交维度"，两者不是替代关系，而是互补。**
capiv2 改的是**外部契约 / ABI**（面向使用者的接口层）；
`tensor-decouple-device-memory` 改的是**内核运行时内存模型**（面向内部的性能层）。

### 8.1 它做了什么（依据该分支源码/提交）

以 `0d60d0e refactor(tensor): decouple device memory allocation from core Tensor to backends`
为核心的一连串重构，叠加大量后处理 / SOPHGO / TRT / insightface / benchmark 优化：

- **`Tensor` 只负责 CPU 内存**：`MemoryBlock` 在 `Device::GPU/TPU` 下直接抛异常，
  明确"Tensor 仅分配 CPU 内存；设备内存由各后端自持"。
- **零拷贝包装**：`Tensor::from_external_memory(data, shape, dtype, deleter, device)`
  用外部指针包装设备内存（用于 GPU/TPU 设备侧数据），配合 `copy_from_extern_buffer`。
- **移除核心 Tensor 里的 `concat` / `softmax` 算子**（`6676654`），把算子下沉/外包，
  进一步瘦身核心数据结构。
- 配套做了大量**性能与正确性**工作：修复 det 双重 sigmoid 的 NMS bug
  （det 463ms→2.4ms、lpr 9.8s→3.7ms）、批量人脸子模型推理、TRT/SOPHGO 全模型转换与 benchmark 等。

### 8.2 它与 capiv2 的耦合点（为什么互补）

capiv2 的 `md_model_predict_nv12(..., MDDevice src_device, ...)` 与
`md_image_plane_ptrs`、以及 `md_draw_result` 的"设备侧 NV12 就地绘制 / 按设备分发"
这套能力，正是建立在"设备内存可由调用方/后端自持并零拷贝直通"的内存模型之上。
也就是说：

- **capiv2 提供"干净、稳定、跨语言的 ABI 入口"**；
- **tensor-decouple 提供"设备内存零拷贝、后端自持内存"的运行时能力**；
- 两者合起来，才能让外部语言（C#/Rust）优雅地做 **GPU/TPU 上的零拷贝 NV12 直通推理**。

### 8.3 一句话区分

```
capiv2/绑定      = 接口层重构：让 25+ 模型变成一套稳定 ABI，绑定零冗余      （面向“怎么写/怎么调”）
tensor-decouple = 内核层重构：让设备内存从核心 Tensor 解耦到后端，支持零拷贝 （面向“跑多快/怎么存”）
```

二者发生在不同层次，目标不同，可独立评审、也可协同演进（甚至在 GPU/NPU 场景强耦合）。

---

## 10. 演进建议（综述）

1. **以 capiv2 为稳定契约长期化**：把 `md_capi.h` 当作 ABI 规范版本化，
   版本间只做增量（新增枚举/函数，不改旧签名），保证二进制兼容。
2. **绑定层用"契约生成"来进一步消灭样板**：当前 C#/Rust 已是"写一遍通用外壳"，
   若模型继续增多，可考虑由 `md_capi.h` 或一份 IDL 自动生成 `DllImport` 与 `model_wrapper!` 条目。
3. **接入 tensor-decouple 的内存模型**：让 C#/Rust 的新绑定在其上暴露"设备帧零拷贝
   NV12 直通 + 设备侧就地绘制"的一流体验（二者天然互补）。
4. **保持"每句柄单线程 + 并发各自 create"契约**，配合 Rust 的 `Send/Sync` 声明，
   形成明确、可被类型系统检查的并发模型。

---

## 附录 A：参考文件

- `capi/md_capi.h`（431 行）、`capi/md_capi.cpp`（2204 行）—— 新契约本体
- `capi/`（61 文件 / 约 4581 行，`main` 分支）—— 原契约
- `csharp/ModelDeploy/NativeMethods.cs`、`types_internal_c.cs`、`V2/*.cs` —— C# 新绑定
- `csharp/ModelDeploy/vision/**（main）` —— C# 旧绑定对照
- `rust/modeldeploy/src/{ffi,model,runtime,image,types,error,lib}.rs` —— Rust 绑定
- `tensor-decouple-device-memory` 分支 `csrc/core/tensor.{h,cpp}` —— 内存模型重构对照

## 附录 B：术语速查

| 中文 | 英文 | 含义 |
|------|------|------|
| 不透明句柄 | Opaque Handle | 只给指针、不暴露内部结构的资源引用 |
| 门面 | Facade | 用少数入口收敛复杂子系统 |
| 防腐层 | Anti-Corruption Layer | 内核与外部之间的翻译/隔离带 |
| 稳定 ABI | Stable ABI | 不随实现变化而破裂的二进制接口 |
| 指针隐藏实现 | PIMPL | 用指针隐蔽实现的惯用法 |
| RAII | Resource Acquisition Is Initialization | 资源随对象生命周期自动管理 |
| 安全外壳绑定 | Safe wrapper over unsafe FFI | 把 unsafe 收敛到边界，对外只露安全 API |
| 零拷贝 | Zero-copy | 跨语言/跨设备传递时避免序列化与拷贝 |
| 单一分发点 | Single Dispatch Point | 同一类操作只有一个统一入口 |

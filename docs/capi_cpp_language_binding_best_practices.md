# capi 封装 C++、供多语言绑定的最佳实践

> 主题：如何用一套**稳定的 C ABI** 把 C++ 推理 SDK 暴露给 C# / Rust / Python 等语言，
> 让绑定层"零冗余、零泄漏、零 UB"。
> 本文不是理论拼盘，而是从本项目 `capiv2`（`capi/md_capi.{h,cpp}`）与原始 `capi/`（v1）
> 的真实工程经验中提炼出的**可落地清单**。随文档配套可参考 `docs/capi_bindings_analysis.md`
> （侧重优劣分析）与本项目源码。

---

## 0. 一句话纲领

> **C++ 内核 + 稳定 C ABI 边界（只暴露不透明句柄与 blittable 数据）+ 各语言薄封装。**
> 所有"不安全"（指针、内存、并发、异常）都收敛在 C 边界内部，
> 各语言绑定只做"拿句柄 → 存进 RAII 对象 → 取 blittable 数组 → 释放"这一件事。

---

## 1. 总体架构：三层，职责不重叠

```
┌─────────────────────────────────────────────────────────────┐
│  第 3 层  语言绑定  C# (ModelDeploy/V2) · Rust (modeldeploy) │ ← 只做 RAII + 类型安全
├─────────────────────────────────────────────────────────────┤
│  第 2 层  稳定 C ABI  capi/md_capi.{h,cpp}                 │ ← 不透明句柄 + 零拷贝 + 统一所有权
├─────────────────────────────────────────────────────────────┤
│  第 1 层  C++ 内核  csrc/（runtime · vision · audio）         │ ← 真正的模型/推理
└─────────────────────────────────────────────────────────────┘
```

三条铁律：
1. **第 1 层绝不直接暴露给语言层**；中间隔一层 C，避免 C++ ABI（rtti/异常/STL）撕裂跨界。
2. **第 2 层是唯一允许出现 `unsafe`/裸指针/`reinterpret_cast` 的地方**。
3. **第 3 层只做映射**，不承担任何内存/并发决策，越薄越好。

---

## 2. C ABI 契约设计核心原则

### 2.1 不透明句柄（Opaque Handle）—— 禁止结构体裸奔

**坑（v1）**：`MDModel{ type; char* model_content; }`、`MDDetectionResults{ size; data; }` 直接暴露，
跨语言侧读写错位即 UB，编译器根本不报。

**最佳实践**：所有资源用**前向声明的空 struct 指针**，调用方只见 `Handle` 不可解引用：

```c
typedef struct md_model_handle*   MDModelHandle;
typedef struct md_result_handle*  MDResultHandle;
```

### 2.2 纯 C99 + 固定导出 + `extern "C"`

**坑**：头文件混入 `bool`/默认参数/引用等 C++ 语法，某些 C 编译器编不过。

**最佳实践**：
- 头文件只写 C99 语法；`bool` 一律用 `int 0/1`；
- 用 `extern "C"` 括住，防 C++ 符号名篡改；
- 导出宏统一（Windows `dllexport/dllimport` / Linux `visibility`）：

```c
#if defined(_WIN32)
#  if defined(MD_CAPI)
#    define MD_CAPI_EXPORT __declspec(dllexport)
#  else
#    define MD_CAPI_EXPORT __declspec(dllimport)
#  endif
#else
#  define MD_CAPI_EXPORT __attribute__((visibility("default")))
#endif
```

### 2.3 单一分发点（Facade）—— 让"模型"变成"数据"

**坑（v1）**：每模型一组函数族（61 文件 / 4581 行），新模型三语各抄一遍。

**最佳实践**：创建/推理/释放各自只有一个入口，按 `ModelKind` 枚举内部分发：

```c
MDStatus md_model_create(MDModelHandle* out, MDModelKind kind,
                         const char* path, const MDOptionHandle opt);
MDStatus md_model_predict(MDModelHandle, MDImageHandle, MDResultHandle* out);
void     md_model_destroy(MDModelHandle);
```

**新增模型 = 加一个枚举 case，绑定层一行不用动。**

### 2.4 统一内存所有权 —— 每个资源"一个 create 配一个 destroy"

**坑（v1）**：`malloc/new[]/strdup/free` 混用，释放函数只 `delete[] data` 漏掉 strdup/new。

**最佳实践**：**所有权一律归库**。输入可"借用"（`from_bgr24` 引用外部内存），但输出/结果
一律库内分配 + 库内释放，并且**嵌套数据也锁在同一个句柄内部**，对外只有一个释放入口：

```c
MDStatus md_model_predict(..., MDResultHandle* out);  // 库分配
void     md_result_destroy(MDResultHandle);            // 唯一释放，嵌套数据同链释放
```

### 2.5 富错误模型 + 线程安全错误通道

**坑（v1）**：每函数返回散落状态码，排查只能靠日志。

**最佳实践**：
- 一个 `MDStatus` 枚举覆盖参数/状态/并发/解码等错误类别；
- 配 `const char* md_get_last_error(void)`，用 **thread_local** 存错误串，避免线程互相污染。

### 2.6 线程模型显式化

**最佳实践**：明确"**每个句柄单线程使用**；并发各自 create 句柄；库内不加锁保证零开销"，
并把"句柄被并发使用"建模为错误（如 `MD_ERR_BUSY`），给 Rust 的 `Send/Sync` 声明提供依据。

### 2.7 blittable 结果 + 数组式 getter（零拷贝）

**坑（v1）**：手写深拷贝把 C++ 结果 `new[]`+`strdup` 成 C 结构，三份拷贝 + 丢精度。

**最佳实践**：
- 固定字段结果用**纯数值 blittable 结构体**（`float/int`，无指针）；
- getter **一次返回库内 vector 的内部指针 + 长度**，调用方"借用"读取，零拷贝；
- 变长数据（关键点/mask/字符串）**不复制**，留在原始 C++ 容器、按索引回传指针。

详见《分析文档》第 6 节的数据流。

### 2.8 数组结果：为什么不用"柔性数组"，而用"数组 getter 指针"

**坑**：把"数量 + 数组"直接塞进一个结构体（即 C99 柔性数组：

```c
typedef struct {
    uint32_t n;
    MDDetectionItem data[];   // 柔性数组：末尾不写长度
} MDDetectionItemsBlob;
```

柔性数组是"头 + 紧贴其后的一段数据"的单块 blob，调用方要手动算总内存一次性分配
（`sizeof(header) + n*sizeof(T)`）再 memcpy。

**为什么跨语言绑定难处理**：
- C# 的 `[MarshalAs(UnmanagedType.ByValArray)]` 必须写死长度，**映射不了长度未知的柔性数组**，
  只能读头 + 手工算偏移（`IntPtr.Add`）+ 手动循环，一个 `sizeof`/padding 算错就踩内存；
- Rust `#[repr(C)]` 里柔性数组只能 `[T; 0]` + 手动偏移 hack，同样脆；
- `sizeof(blob)` 只给头部大小，跨语言方极易把它当全长去拷贝/释放；
- 柔性数组倾向"整块深拷贝/复杂所有权转移"，**无法零拷贝借用**。

**本项目最佳实践：数组不进结构体，而是 getter 的输出参数**——独立的 blittable 结果项 +
数组式 getter：

```c
typedef struct MDDetectionItem { float x,y,w,h; float score; int label_id; } MDDetectionItem;
MDStatus md_result_detection(MDResultHandle h,
                             const MDDetectionItem** items,  // 输出：指向库内 vector 首元素
                             size_t* count);                 // 输出：元素个数
```

实现上（`md_capi.cpp:1295`）：内部把 C++ 结果投影成 `std::vector<MDDetectionItem>`，
getter 一行返回 `p->v.data()` + `p->v.size()`。**数组住在库内，调用方只"借用"只读指针**，
所有权归库，`md_result_destroy` 单链释放。

**三种语言各自怎么读这个数组**：
- **C**：拿 `items`+`count` 就地 `items[k]` 遍历；
- **C#**：`ResultReader.ReadItems<T>` 里
  `Marshal.PtrToStructure<T>(IntPtr.Add(items, i * Marshal.SizeOf<T>()))` 逐项拷成托管数组；
- **Rust**：`std::slice::from_raw_parts(items, n)` 一行切成连续切片，再 `map` 成强类型。

**对比 v1 的 `MDDetectionResults{ size; data; }`（同为"size+指针"）**：思想一致，但 v1 把
`size/data` 当**结构体字段直接暴露**、且由调用方 `new[]`/`strdup` 管所有权、深拷贝丢精度；
capiv2 把它改成**函数输出参数**、数组锁库内 vector、只读借用。同样写法的本质区别在于
"结构体字段（可改、需管所有权）" vs "函数返回值（只读借用、所有权归库）"。

**结论**：柔性数组只适合"一次性传输、长度一次给足、固定/已知"的二进制 blob（如序列化权重、
文件）。"模型结果"这种**高频、要零拷贝、按元素读、每元素还嵌套变长字段**的场景，
本项目用**固定字段数组 getter + 变长字段按索引 getter** 两通道，均不引入柔性数组。

---

## 3. 参数设置工程（本项目的核心经验）

这是"对象参数"最常见也最容易做砸的地方，分三类。

### 3.1 运行时全局选项：Option Handle + 链式 setter（Builder）

**坑**：把每个选项摊平成 C 函数的一大批参数（`f(model, thread, fp16, device, backend, ...)`），
或暴露一个可被改的 struct，改一个字段牵动所有调用点、且不改入参有默认值问题。

**最佳实践**：`option` 是一个**句柄**，内部持有 `RuntimeOption`，用一组 setter 链式设置：
创建默认值（CPU+ORT）→ 按需覆盖 → 传给 `md_model_create`。

```c
MDStatus md_option_create(MDOptionHandle* out);          // 默认 CPU + ORT
void     md_option_set_backend   (MDOptionHandle, MDBackend);
void     md_option_set_device    (MDOptionHandle, MDDevice);
void     md_option_set_cpu_threads(MDOptionHandle, int);
void     md_option_set_fp16      (MDOptionHandle, int);   // int 0/1，不用 bool
void     md_option_set_trt_engine_path(MDOptionHandle, const char*);
void     md_option_destroy(MDOptionHandle);
```

好处：
- **默认值在 `md_option_create` 里一次给足**，调用方只改关心的项；
- 语言层容易做成 fluent（C# `RuntimeOption2.UseOrt().SetDevice(...).SetFp16(true)`）；
- 无参化 `md_model_create` 调用点，新增选项只加 setter、不改既有函数签名（ABI 稳定）。

注意：`md_option_create` 拿到的是"配置快照"，`md_model_create` 里**拷贝一份进模型句柄**
（`mh->opt = opt`，供 clone 复用），随后 option 生命周期独立、可提前释放——语义清晰。

#### 3.1.1 教材级实例：C# 的 `RuntimeOption2` 完整用法手册

下面按**真实源码**逐层拆解（左→右：C# 封装 → P/Invoke 声明 → C 原生实现 → C++ 内核字段），
读完你对"一个运行时选项如何从 C# 一路通到 C++"会有完整画面。

**① 相关真实源码铺底**

`csharp/ModelDeploy/ModelDeploy/V2/BaseModel.cs`（`RuntimeOption2`，约 12–51 行）：

```csharp
public sealed class RuntimeOption2
{
    internal IntPtr Handle { get; private set; }   // 底层 MDOptionHandle
    private bool _ownsHandle;                       // 本实例是否负责释放

    public RuntimeOption2()                         // 默认：CPU + ORT
    {
        md_option_create(out var h);
        Handle = h;
        _ownsHandle = true;
    }

    internal RuntimeOption2(IntPtr existing, bool owns = false) { ... }  // 内部包装用

    public RuntimeOption2 UseOrt()    { md_option_set_backend(Handle, 0); return this; }
    public RuntimeOption2 UseMnn()    { md_option_set_backend(Handle, 1); return this; }
    public RuntimeOption2 UseTrt()    { md_option_set_backend(Handle, 2); return this; }
    public RuntimeOption2 UseSophgo() { md_option_set_backend(Handle, 3); return this; }

    public RuntimeOption2 SetDevice(Device d)        { md_option_set_device(Handle, (int)d); return this; }
    public RuntimeOption2 SetCpuThreads(int n)       { md_option_set_cpu_threads(Handle, n); return this; }
    public RuntimeOption2 SetFp16(bool enable)       { md_option_set_fp16(Handle, enable ? 1 : 0); return this; }
    public RuntimeOption2 SetTrtEnginePath(string p) { md_option_set_trt_engine_path(Handle, p); return this; }

    public void Dispose()
    {
        if (_ownsHandle && Handle != IntPtr.Zero)
        {
            md_option_destroy(Handle);
            Handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }
}
```

`csharp/ModelDeploy/ModelDeploy/NativeMethods.cs`（P/Invoke，约 28–51 行）：

```csharp
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern MDStatus md_option_create(out IntPtr handle);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_destroy(IntPtr handle);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_set_device(IntPtr handle, int device);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_set_backend(IntPtr handle, int backend);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_set_cpu_threads(IntPtr handle, int n);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_set_fp16(IntPtr handle, int enable);
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
internal static extern void     md_option_set_trt_engine_path(IntPtr handle, string path);
```

`capi/md_capi.cpp`（原生实现，约 181–227 行）：

```c
struct md_option_handle {
    modeldeploy::RuntimeOption opt;     // 真正的运行时配置
    bool backend_explicit = false;      // 是否显式指定过后端
};

MDStatus md_option_create(MDOptionHandle* out) { *out = new md_option_handle(); return MD_OK; }
void md_option_destroy(MDOptionHandle h) { delete static_cast<md_option_handle*>(h); }

void md_option_set_device(MDOptionHandle h, MDDevice d) {
    switch (d) {
        case MD_DEV_CPU: o->opt.use_cpu(); break;
        case MD_DEV_GPU: o->opt.use_gpu(0); break;
        case MD_DEV_TPU: o->opt.use_sophgo_backend(0); break;
    }
}
void md_option_set_backend(MDOptionHandle h, MDBackend b) {
    switch (b) { /* ORT/MNN/TRT/SOPHGO 对应 use_*_backend() */ }
    o->backend_explicit = true;
}
void md_option_set_cpu_threads(MDOptionHandle h, int n)  { o->opt.set_cpu_thread_num(n); }
void md_option_set_fp16(MDOptionHandle h, int enable)    { o->opt.enable_fp16 = enable != 0; }
void md_option_set_trt_engine_path(MDOptionHandle h, const char* path) {
    o->opt.ort_option.trt_engine_cache_path = path ? path : "";
}
```

**② 枚举对照表（C# ↔ C，务必对齐，否则 `(int)` 传错就点错后端/设备）**

| C# `Device` | 值 | C `MD_DEVICE` | 含义 |
|-------------|----|---------------|------|
| `CPU`     | 0 | `MD_DEV_CPU`     | 纯 CPU |
| `GPU`     | 1 | `MD_DEV_GPU`     | CUDA GPU（`use_gpu(0)`，设备 0） |
| `TPU`     | 2 | `MD_DEV_TPU`     | 算能 Sophgo TPU（`use_sophgo_backend(0)`） |
| `OPENCL`  | 3 | `MD_DEV_OPENCL`  | OpenCL（当前设备分发未处理，见注意） |
| `VULKAN`  | 4 | `MD_DEV_VULKAN`  | Vulkan（同上） |

| C# `Backend` | 值 | C `MDBACKEND` | 对应 C++ `use_*_backend()` |
|--------------|----|---------------|----------------------------|
| `ORT`    | 0 | `MD_BK_ORT`    | `use_ort_backend()` |
| `MNN`    | 1 | `MD_BK_MNN`    | `use_mnn_backend()` |
| `TRT`    | 2 | `MD_BK_TRT`    | `use_trt_backend()` |
| `SOPHGO` | 3 | `MD_BK_SOPHGO` | `use_sophgo_backend(0)` |

> 注意：`Device` 枚举带 `OPENCL=3`/`VULKAN=4`，但原生 `md_option_set_device` 的 switch **只处理**
> `CPU/GPU/TPU`，传 `OPENCL/VULKAN` 会落到 `default: break`（静默忽略）。即枚举比实现"宽"，
> 调用方别以为选了 OpenCL 就真启用了 OpenCL。

**③ 方法速查表（一条龙：C# 方法 → 原生函数 → C++ 字段 → 要点）**

| `RuntimeOption2` 方法 | 签名 | 原生调用 | 底层 C++ | 默认值 | 核心要点 |
|----------------------|------|----------|----------|--------|----------|
| 构造 | `RuntimeOption2()` | `md_option_create` | `new md_option_handle()` → 默认 `RuntimeOption()` | CPU + ORT | `ownsHandle=true`；默认对象即"合理可用"配置 |
| `UseOrt`   | `() → RuntimeOption2` | `set_backend(0)` | `use_ort_backend()` | 后端 = ORT | 返回 `this`，可链式 |
| `UseMnn`   | `() → RuntimeOption2` | `set_backend(1)` | `use_mnn_backend()` | — | 同上 |
| `UseTrt`   | `() → RuntimeOption2` | `set_backend(2)` | `use_trt_backend()` | — | 通常还要 `SetTrtEnginePath(...)` 指向预生成的 `.engine` |
| `UseSophgo`| `() → RuntimeOption2` | `set_backend(3)` | `use_sophgo_backend(0)` | — | 配 `SetDevice(Device.TPU)` 更明确 |
| `SetDevice`| `(Device) → RuntimeOption2` | `set_device((int)d)` | `use_cpu/use_gpu(0)/use_sophgo_backend(0)` | CPU | 只认 0/1/2 |
| `SetCpuThreads`| `(int) → RuntimeOption2` | `set_cpu_threads(n)` | `set_cpu_thread_num(n)` | 库默认 | 并发各自句柄时按需调 |
| `SetFp16`  | `(bool) → RuntimeOption2` | `set_fp16(enable?1:0)` | `enable_fp16 = enable` | false | **C# 是 bool，但底层用 int 0/1 传递**（见坑） |
| `SetTrtEnginePath`| `(string) → RuntimeOption2` | `set_trt_engine_path(p)` | `ort_option.trt_engine_cache_path = p` | 空 | 传 `null` 也安全（原生判空） |
| `Dispose`  | `()` | `md_option_destroy` | `delete md_option_handle` | — | 只释放 `ownsHandle` 的；`GC.SuppressFinalize` |

**④ 生命周期 · 所有权 · 快照（最容易理解的 3 件事）**

1. **所有权**：只有 `new RuntimeOption2()`（`ownsHandle=true`）负责释放；工厂/内部包装出来的
   （`ownsHandle=false`）不释放——靠 `_ownsHandle` 标志区分，避免双释放。
2. **快照语义**：`RuntimeOption2` 传进构造函数后，`BaseModel` 在 `md_model_create` 里把
   `opt` **拷贝一份**进模型句柄（`mh->opt = opt`）。所以：
   - 模型拿到的是"当时的配置快照"，之后你再改 option **不影响已创建的模型**；
   - option 用完即可 `using` 释放，模型不受影响。
   这就是为什么 `BaseModel` 构造函数里能放心 `finally { ownedOpt?.Dispose(); }`。
3. **推荐写法**：`RuntimeOption2` 只在创建模型时用一次，`using` 或构造函数自动释放，
   不必长期持有。它与模型的配置是"值语义"（拷贝），不是"共享状态"。

**⑤ 典型用法示例（组合后端）**

```csharp
// CPU + ORT（默认），仅调线程数
using (var opt = new RuntimeOption2().SetCpuThreads(4))
using (var det = new UltralyticsDet2("yolo.onnx", opt))
    ...

// GPU + TensorRT（预生成 .engine）
using (var opt = new RuntimeOption2().SetDevice(Device.GPU).UseTrt().SetTrtEnginePath("yolo.engine"))
using (var det = new UltralyticsDet2("yolo.onnx", opt))
    ...

// 算能 TPU
using (var opt = new RuntimeOption2().SetDevice(Device.TPU).UseSophgo())
using (var det = new UltralyticsDet2("yolo.bmodel", opt))
    ...
```

**⑥ 常见坑与排查（结合真实实现）**

- **`SetFp16(bool)` 的 int 陷阱**：C API 用 `int enable`（0/1）、不用 C `bool`——这是 FFI 跨语言
  的通用纪律（C 无标准 bool 布局，且部分语言映射不直观）。C# 封装里已经 `enable ? 1 : 0` 转换，
  你在 C# 侧无需担心，但**写别的绑定（如直接用 P/Invoke）时务必转 int**。
- **`SetDevice(OPENCL/VULKAN)` 静默无效**：原生 switch 只处理 0/1/2（见枚举表注意）。
- **`UseTrt()` 不带 engine 路径**：TRT 后端通常需要 `md_option_set_trt_engine_path` 指向已用
  `trtexec` 生成的 `.engine`（在线从 ONNX 建 engine 慢）。忘了设路径，TRT 可能退化为重建或失败。
- **选项是线程本地的"创建期"数据**：`RuntimeOption2` 只在 `create` 阶段生效，不是运行时热更；
  改并发/线程数要在 **create 前**完成。
- **`backend_explicit` 字段**：原生记录"是否显式指定过后端"，供未来校验"用户指定后端与
  模型扩展名推断后端冲突"时参考（当前仅记录、未强制报警，属预留扩展点）。

### 3.2 模型输入 / 预处理参数：按子模型隔离

**坑**：pipeline 模型（OCR/LPR/行人属性）有多个子模型，各自的输入尺寸/预处理不同，
用一个笼统"输入尺寸"会串台。

**最佳实践**：拆成独立入口，各自作用到对应子模型：

```c
MDStatus md_model_set_input_size  (MDModelHandle, int w, int h);  // 检测/主模型
MDStatus md_model_set_cls_input_size(MDModelHandle, int w, int h); // 分类子模型（PedAttr）
```

**规则**：**一个 setter 只改一个明确子项**；命名带子模型归属；不支持的 kind 返回
`MD_ERR_UNSUPPORTED_TYPE` 而不是静默。最初只暴露必需参数即可，宁可少不可乱。

### 3.3 推理 / 后处理 / 绘制参数：聚合结构体 + 可空缺省

**坑**：阈值/字体/透明度/是否保存/类别名映射……若是每个都摊成一个参数，函数签名会
多达 7~8 个实参且难看（这正是 v1 `md_draw_detection_result(image, res, threshold, font_path,
font_size, alpha, save_result)` 的样子）。

**最佳实践**：把**语义内聚的一组只读参数**打包成一个聚合结构体，允许传 `NULL` 走库内默认：

```c
typedef struct MDDrawOptions {
    double threshold;               // 默认 0.5
    const MDLabelItem* label_map;   // 类别名映射，可 NULL
    size_t label_map_size;
    const char* font_path;          // 可 NULL
    int font_size;                  // 默认 14
    double alpha;                   // 默认 0.15
    int save_result;                // int 0/1
} MDDrawOptions;
MDStatus md_draw_result(MDImageHandle, MDResultHandle, const MDDrawOptions* opt);
```

规则：(a) 聚合结构体只放**读取型参数**，**不放指针所有权**（所有权归库/调用方，见 2.4）；
(b) `opt == NULL` 时用库内 `default_draw_options()`，给调用方零配置即用；
(c) 结构体字段全 `blittable`（或显式指针字段+size），避免跨语言对齐坑。

### 3.4 多子模型路径：分隔符串联

**坑**：OCR 要 det/cls/rec/字典 4 个文件、TTS 要 7 个片段，若每个都一个参数则
`md_model_create` 参数位数不固定。

**最佳实践**：用**一个字符串 + 约定分隔符**串联，并显式校验段数：

```c
// OCR:  det.onnx|cls.onnx|rec.onnx|dict.txt
// TTS:  model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir
MDStatus md_model_create(..., const char* path, ...);   // '|' 分隔
```
在 `md_model_create` 里 `need_parts(n, "ocr")` 检查段数，缺段直接报错，别等模型内部哑火。

### 3.5 图像输入形态：多种构造器 + 明确"借用 vs 自有"

**最佳实践**：图像输入用一组命名构造器表达格式，并**用注释/命名说清内存归属**：

```c
MDStatus md_image_from_file  (MDImageHandle* out, const char* path); // 库解码 + 持有
MDStatus md_image_from_bgr24 (MDImageHandle* out, const void* bgr, int w, int h); // 借用，不拷贝
MDStatus md_image_from_nv12  (MDImageHandle* out, const void* y, const void* uv,
                              int w, int h, int sy, int suv, MDDevice src); // 零拷贝直通
MDStatus md_image_from_encoded(MDImageHandle* out, const void* bytes, size_t n);
void     md_image_destroy(MDImageHandle);
```

凡是"借用"的接口，必须**显式声明调用方保证生命周期**（如头文件注释），这也是 2.4 的补充。

---

## 4. 结果返回模式

已在《分析文档》第 6 节详述，这里给"最佳实践清单"：
- 列表结果：`md_result_xxx(handle, const T** items, size_t* count)` 一次给 blittable 数组指针+长度；
- 变长项：`md_result_xxx(handle, i, const T** ptr, size_t* n)` 按索引给内部指针；
- 单值结果（age/gender/整图掩码）：直接写回 out 标量或整体给指针 + 宽高；
- 每类结果用 `MDResultKind` 标记，getter 校验 kind，类型不符返回 `INVALID_ARGUMENT`；
- 所有权：`md_result_destroy` 一次释放整条链（origin 单链，见 6 节）。

设计关键：**让"固定字段一次取"与"变长按索引取"分开**，既保住 blittable 零拷贝，
又不会因为把字符串塞进结构体而破坏布局。

---

## 5. 各语言绑定最佳实践

### 5.1 C#（.NET / netstandard2.0）
- **全部 `DllImport` 集中**到一个 `partial class NativeMethods`，CallingConvention 用 `Cdecl`；
- 句柄一律 `IntPtr`；`out IntPtr` 接库内分配句柄；
- 结果项结构体 `[StructLayout(LayoutKind.Sequential)]` 对应 blittable；
- 把原生句柄包进 **`IDisposable` + 析构 + `GC.SuppressFinalize`** 类（RAII 化），
  如 `BaseModel`、`Prediction<T>`、`RuntimeOption2`；
- 用**泛型读取器**（`ResultReader.ReadItems<T>` + `Marshal.PtrToStructure`）一次搞定所有模型读取；
- 选项做 **fluent 链式**（`RuntimeOption2.UseOrt().SetFp16(true)`）；
- 跨平台：`size_t` 用 `UIntPtr`；变长指针用 `IntPtr`；字符串统一 UTF-8 手动编解码。

### 5.2 Rust
- `extern "C"` 声明（`ffi.rs`）与 C 头 1:1；
- **unsafe 收敛到边界**：只在 ffi.rs/自动封装内 `unsafe`，对外只露 `Result<_, MdError>`；
- **`Drop` 实现 RAII**，句柄随所有权自动 `md_*_destroy`，杜绝泄漏；
- 用**宏 + trait** 批量生成模型包装（`model_wrapper!` + `ResultType`），一行一个模型；
- 变长数据 `std::slice::from_raw_parts` 零拷贝切边再 `map` 成强类型；
- 声明 `unsafe impl Send/Sync`（配合单线程句柄契约），让并发正确性交类型系统。

### 5.3 通理
- 语言层**不做**任何内存/并发决策，只做映射；
- 绑定层应有**与原生一一对应的单元/集成测试**（本项目 `ModelDeployUnitTest`/`integration_test.rs`）。

---

## 6. 本项目踩过的坑（v1 教训 → v2 解法 → 遗留）

| # | 坑（v1） | 后果 | v2 解法 |
|---|----------|------|---------|
| 1 | 结构体裸奔、字段可改 | 跨语言读写 UB | 不透明句柄（2.1） |
| 2 | 所有权 `malloc/new/strdup/free` 混用；free 只 `delete[] data` | 嵌套指针泄漏 | 统一 create/destroy + 单链 origin（2.4） |
| 3 | 深拷贝成 C 结构 + `float→int` | 三份拷贝 + 丢坐标精度 | blittable + 数组 getter 零拷贝（2.7） |
| 4 | 强耦合内核类型（转换函数写死 `std::vector<T>`） | 内核改字段→绑定全崩 | 投影容器 + `raw_result<T>` 多态批配 |
| 5 | 手写逐字段循环（`if(empty){...;return;}` 应为 continue） | 首个空实例吞掉剩余结果 | 由统一投影逻辑/单测覆盖 |
| 6 | C++ 异常直接跨 C 边界 | 跨界 UB/崩溃 | C 边界内 `try/catch` 转成 `MDStatus`+错误串 |
| 7 | 线程模型隐含 | 并发写同一句柄 UB | "每句柄单线程+MD_ERR_BUSY"显式契约 |
| 8 | 深层函数参数摊平 | 签名难看、难扩展 | 聚合结构体 + 可空缺省（3.3） |

### 遗留小坑与对策
- **重复调用同一数组 getter**：getter 以 `static_cast<ResultData<T>*>(rh->data)` 开头，
  假设 data 仍是原始结果；若同一句柄重复调用同一 getter 会类型不符读错内存。
  → 对策：让数组 getter 也走 `raw_result<T>(rh)` 判类型（与 face/lpr 一致）。
- **设备帧 CPU 绘制 fallback 陷阱**：NV12 设备帧的 y()/uv() 指向设备内存，若设备后端没启用
  而工厂回退到 CPU 后端，用宿主指针写入会越界/UB。
  → 对策（已实现）：`md_draw_result` 里按 `frame.device()` 精确匹配期望后端
  （`dynamic_cast<Cuda/SophgoProcessorBackend*>`），匹配不到就**拒绝**而非静默回退。

### 其它跨平台/绑定硬规则（通用于一切 C FFI）
- **C++ 异常绝不跨 ABI**：所有可能抛异常的入口在边界内 `try/catch` 并转成错误码；
- **`bool`→`int 0/1`；`size_t`→C# `UIntPtr`；浮点存量默认 float，必要时显式 double**；
- **字符串统一 UTF-8 + NUL 结尾**，所有权归库（或调用方），注释写明；
- **结构体对齐**：blittable 结构体字段顺序与《头文件》严格一致，跨平台验证 padding；
- **ABI 版本化 + 只增不改**：新能力加新枚举/新函数，永不改旧签名与旧布局（详见第 8 节）；
- **MSVC**：SDK 需 `/utf-8`，否则源码内中文字符串常量会乱（根 CMakeLists 已全局设置）。

---

## 7. 端到端最小范例（检测模型 → C# / Rust）

### 7.1 C 侧头文件骨架

```c
extern "C" {
MD_CAPI_EXPORT MDStatus md_option_create(MDOptionHandle* out);
MD_CAPI_EXPORT void     md_option_set_backend(MDOptionHandle, MDBackend);
MD_CAPI_EXPORT void     md_option_destroy(MDOptionHandle);

MD_CAPI_EXPORT MDStatus md_image_from_file(MDImageHandle* out, const char* path);
MD_CAPI_EXPORT void     md_image_destroy(MDImageHandle);

MD_CAPI_EXPORT MDStatus md_model_create(MDModelHandle* out, MDModelKind kind,
                                        const char* path, const MDOptionHandle opt);
MD_CAPI_EXPORT MDStatus md_model_predict(MDModelHandle, MDImageHandle, MDResultHandle* out);
MD_CAPI_EXPORT void     md_model_destroy(MDModelHandle);

MD_CAPI_EXPORT MDStatus md_result_detection(MDResultHandle, const MDDetectionItem** items, size_t* count);
MD_CAPI_EXPORT void     md_result_destroy(MDResultHandle);
}
```

### 7.2 C# 绑定片段

```csharp
using var opt = new RuntimeOption2().UseOrt().SetDevice(Device.CPU);
using var img = VisionImage.FromFile("a.jpg");
using var model = new UltralyticsDet2("model.onnx", opt);
using var pred = model.Predict(img);            // Prediction<DetectionResult>
foreach (var d in pred) Console.WriteLine($"{d.LabelId} @ {d.Box} {d.Score:F3}");
pred.Draw(img /*canvas*/, DrawOptions.Default); // 句柄直达 C++ vis_*
```

### 7.3 Rust 绑定片段

```rust
let opt = RuntimeOption::default().use_ort();
let img = Image::from_file("a.jpg")?;
let model = UltralyticsDet::new("model.onnx", &opt)?;
let dets = model.predict(&img)?;                 // Vec<Detection>
for d in &dets { println!("{} @ {:?} {:.3}", d.label_id, d.rect, d.score); }
// model/img 超出作用域，Drop 自动 md_*_destroy
```

---

## 8. 工程化收尾

1. **ABI 版本化 + 只增不改**：`md_capi.h` 当规范版本管理；新增用新枚举/新符号，不改旧签名；
   给 `MDStatus`/`MDModelKind` 留 `XXX_COUNT` 哨兵便于迭代。
2. **契约可生成**：既然"模型=枚举(数据)"，可进一步由 `md_capi.h` 或一份 IDL 自动生成
   C# `DllImport` 与 Rust `model_wrapper!` 条目，把绑定层也变成"生成的而非手写的"。
3. **测试三层各一层**：原生 `ctest`(Catch2) + 每种绑定的单元/集成测试；
   至少覆盖：默认值、内存泄漏（反复 predict/destroy）、并发各自句柄、异常转错误码、ABI 迁移。
4. **文档化"所有权 + 线程"契约**：把"谁分配谁释放 / 每句柄单线程"写进头文件注释，比任何口头约定都可靠。

---

## 附录：一句话原则速记

- 结构体不裸奔 → 不透明句柄。
- 谁分配谁释放 → 统一 create/destroy，嵌套数据锁句柄内。
- C++ 的东西不跨边界 → 只过 blittable 与指针。
- 参数按语义聚合 → Option(链式) + 子模型 setter + 可空聚合结构体。
- 结果零拷贝 → 固定字段数组 getter + 变长按索引 getter。
- 异常/并发/内存/unsafe → 全部收敛在 C 边界内。
- 绑定层越薄越好 → C# RAII + fasade；Rust 安全 wrapper + Drop + 宏。

# capiv2 风险登记表（Risk Register）

> 登记日期：2026-08-18
> 来源：`capi2`（`capi2/md_capi.h` / `capi2/md_capi.cpp`）设计分析，逐行代码核查得出。
> 状态词表：`未处理` / `处理中` / `已修复` / `接受风险`
> 级别定义：
> - **高**：现有用法即可触发，后果为内存破坏 / 未定义行为；
> - **中**：当前潜伏，未来改动或特定调用序列下可触发，后果为泄漏 / 维护风险；
> - **低**：语义 / 命名 / 提示类问题，不影响正确性；
> - **信息**：设计层面的可选增强项。
>
> 配套文档：[capi2_bindings_analysis.md](capi2_bindings_analysis.md)（设计分析）、
> [capi_cpp_language_binding_best_practices.md](capi_cpp_language_binding_best_practices.md)（最佳实践手册）。

---

## 1. 总览表

| ID | 级别 | 问题 | 位置 | 状态 |
|----|------|------|------|------|
| R1 | **高** | 同一数组 getter 二次调用 = 未定义行为 | `md_capi.cpp` 9 处 `static_cast<ResultData<...>>(rh->data)`：1542 / 1562 / 1585 / 1617 / 1639 / 1746 / 1757 / 1900 / 1948 | **已修复（2026-08-18）** |
| R2 | 中 | `rh->data` 语义类型随时间漂移（`ResultData*` → `ProjectedResult*`），每个 getter 都要"猜"当前状态 | `md_capi.cpp:1552`（`rh->data = p`） | 未处理 |
| R3 | 中 | `ProjectedResult` 以裸指针持有 `origin`，`new` 之后若在赋值前插入错误 return 则（单）泄漏 | `md_capi.cpp:1543–1552` 模式 | 处理中（数组 getter 已消窗；face/lpr 保留旧模式但无 UB） |
| R4 | 中 | 依赖 RTTI（`dynamic_cast`），而句柄内 `kind` 枚举已含相同信息 | `raw_result` `md_capi.cpp:136`；2159 / 2165 / 2171 | 未处理（可选） |
| R5 | 低 | `md_result_count` 投影后读投影容器 count（当前数值碰巧相等，语义已漂移） | `md_capi.cpp:1158` | **已修复（2026-08-18）** |
| R6 | 低 | 命名拥挤：`ResultData` / `SingleResult` / `ProjectedResult` / `md_result_handle` 四者共享 "Result" 语义 | `md_capi.cpp:100–132` | 未处理 |
| R7 | 低 | `MD_DEV_OPENCL` / `MD_DEV_VULKAN` 枚举存在但 `md_option_set_device` 的 switch 未实现，传值静默忽略 | `md_capi.cpp` `md_option_apply_device_` 分发 | **已修复（2026-08-18）** |
| R8 | 低 | `set_error_fmt` 错误串截断于 512 字节 | `md_capi.cpp:91–98` | **已修复（2026-08-18）** |
| R9 | 信息 | 所有权仅靠头文件注释表达，无签名级标注（对比 FastDeploy 的 `__fd_give/take/keep`） | `md_capi.h` 全文 | 未处理（可选增强） |

---

## 2. 逐条详情

### R1（高）：同一数组 getter 二次调用 = 未定义行为

**机制**

```cpp
// 以 md_result_detection（md_capi.cpp:1538）为例
auto* d = static_cast<ResultData<DetectionResult>*>(rh->data);  // 假定 data 是"原始容器"
auto* p = new ProjectedResult<MDDetectionItem>();
// ...投影...
p->origin = d;
rh->data = p;      // ← 第一次数组 getter 之后，data 已变成 ProjectedResult*
```

第二次调用同一 getter 时，`static_cast<ResultData<DetectionResult>*>(rh->data)`
把一个 `ProjectedResult*` 强转成 `ResultData<DetectionResult>*` 去读 `d->v`：
两个类型的成员布局不同（`ResultData::v` 的位置上是 `ProjectedResult` 的
vtable 指针 / `v` / `origin`），读取结果是**未定义行为**（读到野值、越界、崩溃，
"看起来正常"只是运气）。

**受影响 getter（9 处，均为数组式 getter）**

| 行号 | 函数 |
|------|------|
| 1542 | `md_result_detection` |
| 1562 | `md_result_classification` |
| 1585 | `md_result_pose` |
| 1617 | `md_result_obb` |
| 1639 | `md_result_instance_seg` |
| 1746 | `md_result_face_rec` |
| 1757 | `md_result_insightface` |
| 1900 | `md_result_attribute` |
| 1948 | `md_result_age` 系（`ResultData<int>*`） |

**触发条件**：同一 `MDResultHandle` 上，对同一数组 getter 调用第 2 次。
例：`md_result_detection(res, &a, &n1)` 之后再 `md_result_detection(res, &b, &n2)`。

**最小复现思路**（不写实现，仅思路）：
1. predict 出一个 detection 结果；
2. 连续调两次 `md_result_detection`；
3. 在 ASan/MSVC 下运行，或直接对比两次返回的 `count`/首项——第 2 次大概率异常。

**为什么 face/lpr 没这个 bug**：`md_result_face`（1698）/`md_result_lpr`（1827）
走的是 `raw_result<T>()`（`md_capi.cpp:136`），它用 `dynamic_cast` 区分
"原始容器 / 已投影的 origin"两种状态——**正确的设施已经存在，只是数组 getter 没统一使用**。

**修复方向**：
- 短期：把 9 处数组 getter 的取容器逻辑统一改为经 `raw_result<T>()` + "投影已存在则复用"（复用 face/lpr 模式）；
- 长期：见 R2（handle 改双槽位后此问题从根上消失）。

**修复记录（2026-08-18）**：已按短期方案落地，`tests/test_capi.cpp` 新增回归
`capi2 result getters are idempotent and standalone-safe`（[model]，本地已跑通）：
- 新增 `project_cached<Src,Dst>(rh, fill)` 助手：已缓存同类型投影则幂等复用，否则经 `raw_result` 解析真实 origin 构建；
- 7 个数组 getter（detection / classification / pose / obb / instance_seg / insightface / attribute）改用之；
- "依赖型" getter（pose keypoints / iseg mask / insightface kps·embedding·pose / attr_scores / face_kps / plate / lpr_keypoints）改为直接读 `raw_result`，不再假设必须先调数组 getter；
- face_rec_embedding / spoof 亦改 `raw_result`（防御）。
- **实测放大**：修复前 detection 二次调用即取到野值（score=0、label_id≈1063892810），classification 二次调用直接 **SIGSEGV**——非"碰巧正常"，是真实崩溃；修复后全绿（146 断言 / 12 capi 用例 + 全量非模型 1094/113）。
- 表格中"1948"实际为 `md_result_spoof`（`ResultData<int>*`）；`md_result_age`（1928）用的是 `SingleResult<int>`，不在本高危之列。

### R2（中）：`rh->data` 的语义类型随时间漂移

`void* data` 字段在结果句柄生命周期内会承载三种类型：
`ResultData<T>*`（predict 后）→ `ProjectedResult<Dst>*`（首次数组 getter 后）→ 单值 kind 则是 `SingleResult<T>*`。
每个 getter 都必须"猜"它现在是谁，R1 就是猜错的代价。

**修复方向**：`md_result_handle` 改为双槽位（纯内部重构，C ABI 不变）：

```cpp
struct md_result_handle {
    MDResultKind kind;
    ResultDataBase*      origin    = nullptr;  // predict 时填，终身不变
    ProjectedResultBase* projected = nullptr;  // 可选缓存，惰性生成
    ~md_result_handle() { delete projected; delete origin; }
};
```

数组 getter 永远从 `origin` 读、投影命中缓存则复用；变长 getter 直接读 `origin->v[i]`
（不再绕 `p->origin`）；`md_result_count` 固定读 `origin->count()`。
该重构同时消解 R1 / R3 / R5 / R6。

### R3（中）：`ProjectedResult` 裸指针持有 origin 的泄漏窗口

```cpp
auto* p = new ProjectedResult<MDDetectionItem>();  // 已分配
// ← 若此处在 p->origin=d 前插入错误 return：p 泄漏（单泄漏）
p->origin = d;                                     // 所有权此刻才转移
rh->data = p;
```

**核验修正（2026-08-18）**：原描述"双泄漏（p + origin）"不准确——在插入点
（`new` 与 `p->origin=d` 之间）origin **尚未赋值**，`rh->data` 仍持原 `ResultData`，
由其析构正常释放；实际只漏 `p`（且 `/EHsc` 关闭下无异常路径，OOM 直接 abort）。
风险主要是未来代码插入提前 `return` 造成单泄漏。本次 `project_cached` 已把
数组 getter 的 `new` 移到 `raw_result` 判空之后，且 fill 无错误分支，泄漏窗口已消除；
face/lpr 仍保留"先 `new` 再判空分支"的旧模式，但无 UB，仅理论单泄漏。

**修复方向**：`origin` 改 `std::unique_ptr<ResultDataBase>`；或先挂入 handle、后填充内容。

### R4（中，可选）：RTTI 依赖与 `kind` 信息冗余

`raw_result<T>()` 用 `dynamic_cast` 判型（`md_capi.cpp:136`），2159/2165/2171 的
`sem_seg/depth/ocr` getter 也是 `dynamic_cast`。而句柄里的 `MDResultKind` 已经唯一确定
容器类型（每个 kind ↔ 一个 C++ 结果类型）。

**取舍**：MSVC 下 RTTI 支持良好、可读性好，当前不构成性能问题；
若追求零 RTTI 构建或更小的 .rdata/.pdata，可改为按 `rh->kind` 的 switch 分派强转
（代价：新增 kind 时记得补 case）。

### R5（低）：`md_result_count` 语义漂移

`md_capi.cpp:1158`：`*out = static_cast<ResultDataBase*>(rh->data)->count();`
投影之后这里读的是投影容器的 count。当前投影元素数恒等于原始元素数，**数值碰巧正确**；
语义上"count 应该指原始结果"，随 R2 修复后固定读 `origin->count()`。

**修复记录（2026-08-18）**：新增 `origin_count(rh)` 助手（经 `ProjectedResultBase::origin_ptr()`
读真实 origin），`md_result_count` 改用它，明确读原始结果条数。

### R6（低）：命名拥挤

`ResultData`（原始容器）/ `SingleResult`（单值容器）/ `ProjectedResult`（投影缓存）/
`md_result_handle`（句柄）四个标识符共享 "Result" 一词，且与对外 API 的
`MDResultHandle` 概念纠缠。建议随 R2 重构一并更名（如 `ResultOrigin` / `ResultView`）。

### R7（低）：OPENCL / VULKAN 枚举是"空头支票"

C 头与 C# `Device` 枚举都含 `OPENCL=3` / `VULKAN=4`，但 `md_option_apply_device_`
（`md_capi.cpp:199` 的 switch，由 `md_option_set_device` 触发调用）只处理
`CPU/GPU/TPU`（`default: break`），传 3/4 时 `o->device` 虽被存为 3/4，后端却**静默无操作**。
调用方以为启用了 OpenCL 实际没有。
**修复方向**：要么实现，要么在 setter 里返回/记录 `MD_ERR_UNSUPPORTED`，
至少头文件注释标注"预留、未实现"。

**修复记录（2026-08-18）**：`md_option_apply_device_` 的 `default` 分支改为
`set_error_fmt("md_option_set_device: device %d is reserved/not implemented")`，
不再静默忽略；`md_capi.h` 的 `MD_DEV_OPENCL/VULKAN` 注释标注"预留，未实现"。

### R8（低）：错误串 512 字节截断

`set_error_fmt`（`md_capi.cpp:91–98`）用 `char buf[512]` + `vsnprintf`，
超长消息（如含长模型路径）被静默截断。低概率但真实存在。
**修复方向**：改 `std::string` 组装（thread_local 本身就是 string）。

**修复记录（2026-08-18）**：`set_error_fmt` 先 `vsnprintf(nullptr,0,...)` 测长度，
再用 `std::string` 精确容装，长消息不再截断（含长模型路径）。

### R9（信息）：所有权无签名级标注

capiv2 的所有权契约（谁 create 谁 destroy、结果指针是"只读借用"、快照语义）
目前全部靠头文件中文注释 + 绑定层惯例（C# `_ownsHandle`）表达。
业界更系统的做法是在签名上加纯 `#define` 标注宏：

```c
#define MD_GIVE   /* 返回值：新对象，调用方负责 destroy */
#define MD_TAKE   /* 参数：对象所有权移交本函数，之后不得再用 */
#define MD_KEEP   /* 参数：临时借用，函数返回后仍可用 */
```

零运行时成本、不破坏 C99，且可被文档生成工具 / 绑定生成器（如 uniffi/cbindgen 类）
直接消费。**可选增强**，不修也不影响正确性。

---

## 3. 依赖关系与建议处理顺序

```
R2（双槽位重构）─── 一次性消解 ───→ R1、R3、R5、R6
R1 短期热修（统一走 raw_result + 缓存）─── 若不想动结构，可独立先做
R4（去 RTTI）─── 可随 R2 一起做，也可永远不做
R7 / R8 ─── 独立小改
R9 ─── 独立增强
```

**建议**：R1 是当前唯一"现有用法即可触发"的问题，优先级最高；
若暂不做 R2 大重构，至少先做 R1 短期热修。

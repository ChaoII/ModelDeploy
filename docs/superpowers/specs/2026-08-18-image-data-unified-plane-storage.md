# Spec：ImageData 统一平面存储重构

- 日期：2026-08-18
- 分支：`capi-v2`
- 状态：**待审阅**（用户批准后进 plan）
- 配套：登记册 `docs/image_data_issue_register.md`（基线 `f800c03`）、消费者调研 `sdd/unified-storage-consumers.md`
- 目标基点：当前 HEAD `f800c03`

## 1. 背景与目标

现状 `ImageData` 的"深度绑定 cv::Mat"来自三处（登记册根因 #6/#7/#16）：

1. **双表示**：`CpuStorage{cv::Mat}` 与 `PlaneStorage{planes}` 并存，同一对象两种数据源，元数据 / 平面派生 / 操作语义分叉；
2. **Mat 锚点未拆除**：元数据从 mat 派生（`refresh_meta`）、`mutable plane_mat_` 缓存（数据竞争）、`rotate_crop/imshow` 裸 `impl_->mat()`（设备帧无守卫）；
3. **构造入口不收敛**：7 个构造覆盖 3 类场景，`from_raw` 与 `from_device_planes`/`from_bgr24` 重叠，YUV 半成品（I420 枚举有、路径无）。

**目标**：统一为单一平面存储 `{fmt, w, h, ch, device, planes[3], owner}`，`cv::Mat` 仅作"桥"（摄取构造 + `asMat()` 弹出，不成为数据源）；构造族收敛至 5；操作全部经 backenend 分派（设备帧 fast-fail，不建 backend）；YUV 容器可构造自描述、不可构造的操做显式报错。**P0 全随统一存储结构消解或显式守卫。**

**不做（超出范围）**：R2（`md_result_handle` 双槽位）、CUDA 后处理、SOPHGO 后处理、改 C ABI 签名（`md_*` 对外签名保持不变，仅内部改实现）、OpenCV 依赖移除（仍使用，但只作桥）。

## 2. 设计

### §1 统一平面存储

```cpp
// image_data.h（内部 impl）
struct ImageDataImpl {
    MdImageType fmt = MdImageType::PKG_BGR_U8; // 唯一格式源
    int w = 0, h = 0, ch = 1;                  // 真实宽高（不再从 mat 派生）
    Device device = Device::CPU;
    std::array<Plane, 3> planes{};             // plane= {uint8_t* data; int step;}
    std::shared_ptr<void> owner;               // 保活：自分配 buffer / 借用源
    // 派生缓存（可选，仅一致性校验用）：size_t bytes_, element_count_, element_bytes_;
};
```

- **`cv::Mat` 不再是成员**：删除 `CpuStorage{mat}`、`PlaneStorage` 双分支、`mutable plane_mat_`、`materialize_plane_mat()`、`refresh_meta()`（元数据仅构造时填一次，见 §2）。
- **数据源唯一**：所有数据都在 `planes[]`（`plane_count()` = `planes` 有效数）。`owner` 保活 backing：自分配（`w,h,fmt` 构造 / `copy=true`）时持有 `shared_ptr<std::vector<uint8_t>>`；借用（`copy=false` / `from_planes`）时持有调用方传入的 `owner`（可空=调用方保证存活，与现 `keeper` 语义一致但**真实接线**）。
- **asMat = 弹出桥**：仅对 `packed` 格式（`PKG_BGR_U8` / `PKG_RGB_U8`）产出 `cv::Mat(h,w,C, planes[0].data, planes[0].step)` **借用视图**（不拷贝、不缓存）；多平面 YUV 帧 `asMat` 返回 `false` + 置错（调用方须用 `plane(i)`）。
- **plane(i) 保留现值语义**：返回 `{data, step}`；packed `step`=行字节，NV12/NV21 `plane(0)=Y`、`plane(1)=UV`（step 尊重），I420 `plane(0/1/2)=Y/U/V`。
- **高度契约修复（P0-1/#21）**：`h`/`w` 恒为真实值（构造时填）；不再有 `1.5h` mat。`height()`/`width()` 直接读 impl 字段。

### §2 构造族收敛 7→5

| # | 新构造 | 取代 |
|---|--------|------|
| 1 | `ImageData()` 空 | default |
| 2 | `ImageData(int w, int h, MdImageType fmt)` 自分配 CPU buffer，YUV 亦合法（自描述、非空） | `ImageData(w,h,fmt)`（修 #21：YUV 不再空） |
| 3 | `ImageData(const cv::Mat&)` 桥接摄取（借用；`Mat&&` 并入，多一层 `cv::Mat` 不作为存储） | `const Mat&` + `Mat&&` 两个 |
| 4 | `from_raw(uint8_t* data, int w, int h, MdImageType fmt, bool copy=false, Device device=CPU, std::shared_ptr<void> owner={})` 统一提炼：packed 借用/拷贝；YUV 平铺 buffer 拆平面；可携带 device+owner（修 #9：设备指针不再静默标 CPU） | `from_raw`、`from_device_planes` 的借用路径 |
| 5 | `from_planes(const Plane* planes /*≤3*/, const int* steps, MdImageType fmt, int w, int h, Device device, std::shared_ptr<void> owner)` 泛化可表达 NV12/NV21(2p)/I420(3p)/packed(1p) | `from_device_planes`（删除，改名+泛化） |

- `from_bgr24(const uint8_t*,w,h)` → **降为 `from_raw(data,w,h,PKG_BGR_U8,false)` 的内联便捷别名**（5 个测试点，生产无调用）。保留符号避免波及测试/文档。
- 删除：`from_device_planes`（13 处调用点迁移到 `from_planes`）、以 storage 区分的双 Mat 构造。
- **消费兼容**【全部在 spec 审阅时核对，关键：
  - `md_image_from_bgr24`（capi2:285）继续走 `from_raw`（现已是）。✅
  - `md_image_from_nv12`（capi2:321）→ `from_planes(NV12, 2p)`。✅
  - `md_image_from_device_nv12`（capi2）→ `from_planes(NV12, device 参数)`。✅
  - pybind ~45 个 `Mat&&` 构造点 → `ImageData(const cv::Mat&)` 可接（&& 并入）。✅
  - `from_raw(NV12)` 现生产 3 处（batch_scheduler:154 / infer_group:210 / benchmark_yolo_preproc:54）→ 统一到新平面语义（真实 h），下游 `fused_preprocess` 改读 `plane(0/1)`。

### §3 操作全分派 & 设备帧 fast-fail

- **搬入 CPU backend（经 asMat/plane 桥）**：`rotate_crop`（现 ImageData:433）、`cvt_color` 的 `PA2PL`/`PL2PA`（现 :518）。CPU backend 实现用公开 `asMat()`/`plane(i)`，与现有 resize/crop/rotate/cvt_color 一致。
- **`imshow`**：CPU-only 显示——保留在 ImageData，但显式守卫 `device==CPU`（设备帧→置错返回），不搬后端（非可设备化 op）。
- **`clone`**：CPU 帧→按平面深拷贝到新 owner；非 CPU→显式置错（"device frame clone not supported"），不再静默空图（修 #3）。设备深拷贝（TPU 显存再分配）明确 out-of-scope。
- **设备帧 fast-fail 不建 backend（修 #14）**：新增 `static bool VisionProcessorBackend::supports(Device, Op)`（固定映射：CPU→全部图像 op；GPU/TPU→仅 letterbox 预处理与 NV12 绘制，**不含** crop/rotate/resize/cvt_color/rotate_crop）。`ImageData::{crop,rotate,resize,cvt_color,rotate_crop}` 入口：
  ```cpp
  if (!VisionProcessorBackend::supports(device_, OpKind::Resize)) { set_last_error(...); return ImageData(); }
  auto& bk = *backend_for(device_); // 至此才建 backend
  ```
  设备帧 → 第一行就拦下，**不构造 backend**（省掉每次 op 空 `bm_dev_request/free`）。
- **对外语义不变**：CPU 帧全部路径 bit-identical（回归锁定）。

### §4 YUV 容器支持 / 操作拒绝

- **可构造+自描述**：`ImageData(w,h,NV12/NV21/I420)` 有效；`from_raw`/`from_planes` 可承载上述格式；`plane_count()` 正确（2/2/3）；`format()/width()/height()` 正确。修 #10 的半成品 + #21 的空图。
- **操作显式拒绝**（置 `last_error`，不静默错走）：
  - `toCpu`：仅 NV12（及 packt）可消费；NV21/I420 → `"toCpu: unsupported YUV: <fmt>"`。
  - `cvt_color`：仅 `PKG_BGR/RGB` 互转 + 专用 `CVT_NV122PKG_BGR` 可转换；NV21/I420 无专用转换 → `"cvt_color: unsupported YUV"`。
  - `imencode/imwrite/save`（codec）：非单平面 → 显式拒绝（沿用现 `codec_supported`，补 I420/NV21 明确报错）。
- **`md_image_from_yuv420p` 处理（spec 决策点）**：现绕过 ImageData（`md_capi.cpp:406` 裸 cv + cvtColor）。方案：新增专用 `CVT_I4202PKG_BGR`（OpenCV `COLOR_YUV2BGR_I420`，一行），使 `from_planes(I420)+cvt_color(CVT_I4202PKG_BGR)` 可重路由，行为与原手写路径等价（有像素级测试锁定）。默认**采纳**此方案（保持"YUV 操作拒绝"一致性的同时不破坏 yuv420p 功能）；若评审认为新增枚举超范围，回退为保留手写路径。
- NV21：仅容器支持（可构造自描述），无任何转换——显式拒绝。

### §5 P0 消解矩阵

| 登记册 | 状态 | 机制 |
|--------|------|------|
| #1 NV12 height 1.5h vs h | **结构消解** | 统一存储填真实 h/w；`from_raw(NV12)` 不再包 `1.5h` mat |
| #2 plane_mat_ 数据竞争 | **结构消解** | 删除 mutable mat 缓存；asMat 每次产临时视图，无共享可变状态 |
| #3 clone 设备静默空 | **显式守卫** | clone 非 CPU → last_error 返回 |
| #4 rotate_crop/imshow 无守卫 | **显式守卫** | rotate_crop→CPU backend（支持查询）；imshow→device==CPU 判 |
| #5 CPU-only NV12→BGR 单 mat | **结构消解+待单测** | from_raw(NV12) 平面化；补 CPU-only 像素级单测 |
| #6 双表示 | **结构消解** | 单一存储 |
| #7 mat 锚点 | **结构消解** | mat 只作桥；元数据不派生 |
| #8 cvt_color PA2PL（已受控） | 保持 | 迁入 CPU backend 后更统一 |
| #9 from_raw(gpu_ptr) 静默 CPU | **结构消解** | from_raw 带 device+owner |
| #10 I420 半成品 | **显式/结构** | 可构造自描述 + 操作拒绝 |
| #14 每次 op 建 backend | **结构消解** | supports() 前置 fast-fail |
| #16 元数据散落 | **结构消解** | 构造时单点填入 |
| #21 自分配 YUV 空图 | **结构消解** | 平面存储支持 YUV 自分配 |
| #15/#17/#18/#19/#20/#22 | 顺带 | 构造收敛、keeper 接线、去重、待清理项列入计划 |

### §6 C API 图像操作同样必须走 ImageData（新增，用户要求）

C API 里凡"绕过 ImageData、直接在 `hi->data` 上 `cv::Mat`"的图像函数一律不允许，收敛为走 ImageData 桥（`asMat()`/`plane(i)`/方法分派），设备帧/YUV 走与 SDK 一致的 fast-fail 或显式报错：

| 现存裸操作 | 现状 | 收敛方式 |
|---|---|---|
| `md_image_from_rgb24` | `md_capi.cpp:345-350` 裸 `cv::cvtColor(RGB2BGR)` | `from_raw(rgb, PKG_RGB_U8)` + `cvt_color(CVT_RGB2BGR)`（走分派） |
| `md_image_from_yuv420p` | `:406-411` 裸 `cv::cvtColor(I420)` | `from_planes(I420)` + `cvt_color(CVT_I4202PKG_BGR)`（新增枚举，见 §4） |
| `md_image_show` | `:475-482` 裸 `cv::Mat(...,hi->data)` + imshow | `hi->image.asMat(&m)`（packed-only）→ cv::imshow；YUV/设备 → 错误 |
| `md_image_save` | `:490-495` 裸 imwrite | 同上：`asMat` 桥 + 守卫 |
| `md_image_encode` | `:499-509` 裸 imencode | 同上：`asMat` 桥 + 守卫 |
| `md_draw_rect/line/circle` 等原语 | `:2001/2015/2030` 裸 `cv::Mat(...,hi->data)` | `hi->image.asMat(&m)`（CPU packed）→ 绘制；设备帧 → 错误（NV12 绘制走 `md_draw_result` 既有 device 路径） |

- 原则：**C API 图像函数不自己造 `cv::Mat`，统一经 `asMat()` 桥取 CPU packed 视图或 `plane(i)` 取平面**；非满足约束格式 → 显式 `last_error` + 错误码，不静默。
- 已在走 ImageData 的（from_file/from_bgr24/from_nv12/from_encoded/base64/crop/size/plane_ptrs）保持，仅确认无裸 mat 残留。

## 7. 迁移与兼容

- **对外 C ABI 不变**（`md_*` 签名不动）；`ImageData` C++ 公开方法名 `plane(i)/asMat/toCpu/clone/...` 保留；删除 `from_device_planes`/`from_bgr24`（改别名）。
- **必须同步改的消费方**（调研已枚举，spec 审阅时报数）：capi2 5 处 `md_image_*`、6 个 ultralytics 模型后处理器（NV12→BGR 用 plane）、3 个 backend（cpu/cuda/sophgo 的 letterbox/draw 用 plane(0/1)）、pybind、utils/face_align、ppocr(rotate_crop)、batch_scheduler/infer_group/benchmark（from_raw NV12）、test_image_data/test_capi/test_vision_models。
- **验收**（回归基线）：
  - 全量非模型 `~[model]` ≥ 现状 1094/113 全绿；
  - 新增/修订测试覆盖：`from_raw(NV12)` 真实高度、自分配 YUV 非空、`from_planes(I420/NV21)` 自描述、`cvt_color`/`toCpu` 对 NV21/I420 显式拒绝、`CVT_NV122PKG_BGR` 像素等价、CPU-only NV12→BGR 补测、`asMat` packed-only、device 帧 crop/rotate/resize/rotate_crop fast-fail（不建 backend）、`clone` 设备守卫；
  - `from_bgr24`/`md_image_from_bgr24`、`md_image_from_nv12`、`md_image_from_device_nv12`、`md_image_from_yuv420p` 行为逐位一致。

## 8. 实施阶段（进 plan 后细化为 Tasks）

1. Task 1 · 统一存储落地（新 `ImageDataImpl` + 5 构造 + from_bgr24 别名 + asMat packed-only + 删双表示/mat 缓存/refresh_meta）
2. Task 2 · `supports()` 分派前置 + 设备帧 fast-fail；`from_raw` 带 device/owner（修 #9）
3. Task 3 · 迁移 13 处 `from_device_planes`→`from_planes` + `md_image_from_nv12/device_nv12/yuv420p` + NV12/I420 专用转换（CVT_NV122PKG_BGR 保留、新增 CVT_I4202PKG_BGR）
4. Task 4 · rotate_crop/PA2PL/PL2PA 入 CPU backend；imshow/clone 守卫
5. Task 5 · YUV 操作拒绝（toCpu/cvt_color/codec 对 NV21/I420）+ CPU-only NV12→BGR 补测
6. Task 6 · pybind/application/benchmark 消费对齐 + 清理（#15/#17/#18/#19/#20/#22 汇总）
7. Task 7 · 全量回归 + 像素级测试 + 文档（preprocess.md #22、登记册状态更新）

每 Task 走 implementer + reviewer 双代理（沿用本仓库 SDD 模式）。

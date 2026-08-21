# ImageData 问题登记册

- 日期：2026-08-18
- 分支：`capi-v2`
- **审计基线 HEAD：`65baa31`**（统一平面存储改造完成后的当前 HEAD；下文行号均相对此提交；改代码后行号会漂移，复核时先对照基线）
- 审计范围：`csrc/vision/common/image_data.h/.cpp` 及其消费方（processors / capi / application / pybind / tests）
- 审计方法：逐行代码审计 + `git diff e000f75~1..HEAD` 改造前后对照 + 设计 spec（`docs/superpowers/specs/2026-08-17-image-data-multidevice-rewrite-design.md`）逐条核对 + 全仓库消费方 grep
- 严重度定义：
  - **P0 严重**：错误结果 / 崩溃 / 数据竞争（已发生或必然发生）
  - **P1 严重**：API 契约违例 / 架构原则断裂（后续必然踩中）
  - **P2 中等**：过度设计 / 冗余 / 性能浪费
  - **P3 轻微**：死代码 / 命名 / 文档债
- 状态标注（2026-08-18 复核新增）：
  - **受控** = 该子句已加守卫/已修复（代码现状不再成立）
  - **并入** = 与另一条同根，合并到 #X
  - 未标 = 仍成立、待处理

---

## P0 · 严重

### 1. NV12 帧 `height()` 契约违例：同帧两条构造路径报两个高度

- **状态：已解决（2026-08-18 Task 7 复核）**：`from_raw(NV12)` 现自描述为 2 平面且 `height()==真实 h`，单 mat `(1.5h,w)` 表示已移除；`from_device_planes` 已更名 `from_planes`。原 #5（并入）随之消解。
- 位置：`image_data.cpp:132`（`refresh_meta` 读 `mat.rows`） vs `:262`（`from_device_planes` 填真实 `h`）
- 说明：`from_raw(NV12)` 把帧包成 `(1.5h, w)` 单 mat（`:206`），`refresh_meta` 直接读 `mat.rows` → `height() == 1.5h`；`from_device_planes(NV12)` 显式填真实 `h` → `height() == h`。同一逻辑帧、两种构造、两个高度。
- 影响：
  - `to_tensor` / `images_to_tensor` 产出错误 shape `{c, 1.5h, w}`
  - `fused_preprocess`（`cpu_processor_backend.cpp:233`）会把 NV12 帧当 HWC 图读 1.5h 行
  - **生产两条路并存**：`capi/md_capi.cpp:321`（`md_image_from_nv12`）走 `from_device_planes`，`application/infer_group.cpp:210` 走 `from_raw(NV12)`
- **并入 #5（2026-08-18 复核）**：原 #5（CPU-only NV12→BGR 走 `from_raw(NV12)` 单 mat 路径，`convert.cpp:144-145`）与本条同根——都是 `from_raw` 的"设备无关单 mat"表示。保留其"疑似，需实测"待验证：该路径目前无单测覆盖（像素级测试全走 `from_device_planes`），建议补一个 CPU-only NV12→BGR 单测锁定行为后再决定去留。

### 2. `plane_mat_` 数据竞争

- **状态：已解决（2026-08-18 Task 7 复核）**：`mutable cv::Mat plane_mat_` 缓存已删除，统一平面存储改为 `owner`（`shared_ptr<void>`）持有数据，无 per-instance mutable mat 缓存，竞争点消除。
- 位置：`image_data.cpp:58`（`mutable cv::Mat plane_mat_`）、`:71-84`（`materialize_plane_mat`）
- 说明：per-instance mutable 缓存；`ImageData` 浅拷贝共享 `impl_`，多线程共享同一 `from_bgr24` 图并发调用 `asMat` / `imencode` / `rotate_crop` 时，都会对该 `cv::Mat` 做非原子赋值 → UB。
- 备注：注释自称"替代旧 shared static empty，避免并发线程写同一共享对象"，实际是修了一个竞争、引入另一个。

### 3. 设备帧 `clone()` 静默返回空图

- **状态：已解决（2026-08-18 Task 7 复核）**：`clone()` 现对 `device()!=CPU` 显式 `set_last_error("clone: device frame not supported")` 并返回空图，不再静默。
- 位置：`image_data.cpp:177-188`
- 说明：`clone()` 走 `impl_->mat().clone()`，设备帧的 mat 恒为空 → 返回空图，且**不写 last_error**，静默丢数据。
- 影响：设计 spec §3 承诺"clone() 深拷贝到同设备"未实现；capi `md_image_clone`（`md_capi.cpp:396-410`）被迫只支持 CPU BGR。

### 4. `rotate_crop` / `imshow` 无设备守卫

- **状态：已解决（2026-08-18 Task 7 复核）**：`rotate_crop` 已移入 CPU 后端并加设备守卫；`imshow` 已加设备守卫并写 last_error，不再裸碰空 mat 触发 `cv::Exception`。
- 位置：`image_data.cpp:433-487`（`rotate_crop` 直接 `impl_->mat()`）、`:695-702`（`imshow`）
- 说明：设备帧 → 空 mat → `cv::warpPerspective` / `cv::imshow` 抛 `cv::Exception`；本库无异常（部分 TU 未开 `/EHsc`，有 C4530 背景）→ terminate / assert。
- 影响：`rotate_crop` 被 OCR 使用（`ppocr.cpp:119`），当前输入 CPU-only 属潜伏风险。
- 对比：`imencode` / `imwrite` / `cvt_color(PA2PL)` 都有 `codec_supported` 类守卫，唯 `rotate_crop` / `imshow` 两处漏掉。

---

## P1 · 严重（架构断裂）

### 6. 同一逻辑对象双表示（P0-1 的根因）

- **状态：已解决（2026-08-18 Task 7 复核）**：`CpuStorage`/`PlaneStorage` 已合并为单一 `ImageDataImpl{fmt,w,h,ch,device,planes[3],owner,nplanes}`，不再有双表示。
- 位置：`image_data.cpp:35-47`（`CpuStorage` / `PlaneStorage`）
- 说明：CPU NV12/BGR 帧可存为 `CpuStorage`（mat）或 `PlaneStorage`（planes）：两套元数据派生、两套平面派生、两套 `toCpu`/`clone` 分支，语义必然漂移。

### 7. cv::Mat 仍是 CPU "事实数据源"

- **状态：已解决（2026-08-18 Task 7 复核）**：元数据由平面/字段直接派生，`cv::Mat` 降级为输入桥（`ImageData(const Mat&)` 摄取深度拷贝）与 `asMat()` 弹出（纯 packed 借用），不再是内部数据源。
- 位置：`image_data.cpp:35-38`（`CpuStorage{cv::Mat}`）、`:116-138`（`refresh_meta` 从 mat 派生元数据）
- 说明：设计 spec §1 明确"cv::Mat 降级为 CPU Storage 的一种承载"，实现仍锚定 mat：元数据从 mat 派生、imencode / imshow / rotate_crop 直接碰内部 mat、mutable mat 缓存。

### 8. 操作绕过 backend 分派（cvt_color 部分已受控）

- **状态：已解决（2026-08-18 Task 7 复核）**：`rotate_crop`/`cvt_color(PA2PL、PL2PA)` 均已移入 `CpuProcessorBackend`，`cvt_color`/`crop`/`rotate`/`resize` 全部经后端分派并配 `supports(device,op)` 前置 fast-fail；不再有裸 `impl_->mat()` 操作绕过 backend。
- 位置：`image_data.cpp:433-487`（`rotate_crop` 裸 OpenCV）、`:518-575`（`cvt_color` 的 PA2PL/PL2PA 裸 `cv::split`/`cv::merge`）
- 说明：非全部每方法走后端；设备帧在这些操作下应"明确报错、不静默回退"。
- **状态（2026-08-18 复核）**：
  - `cvt_color` 的 PA2PL/PL2PA 已**受控**：`image_data.cpp:521-524` 现已有显式设备守卫（提交 `2f80468`），设备帧返回 `MD_ERR_UNSUPPORTED` 而非裸 `cv::split/merge` 崩。**本条原描述已过期。**
  - `rotate_crop` 裸 `impl_->mat()` 仍成立——与 **#4 同根，并入 #4**，不再单列。

### 9. `from_raw` 接受 YUV，与 `from_device_planes` 重叠；`from_raw(gpu_ptr)` 静默标 CPU

- **状态：已解决（2026-08-18 Task 7 复核）**：`from_device_planes` 已移除（→ `from_planes`），`from_raw` 对 YUV 自描述为 2/3 平面并带 `device` 参数，重叠语义消解。

- 位置：`image_data.cpp:191-216`
- 说明：两个构造入口表达同一借用 YUV 帧（P1-11 的另一半）；`from_raw` 无法区分设备指针，传入 GPU/TPU 指针会被包装成 CPU mat，`device()` 恒报 CPU。

### 10. I420 半成品

- **状态：已解决（2026-08-18 Task 7 复核）**：I420 现为 3 平面自描述，并新增专用 `CVT_I4202PKG_BGR` 后端路径（`cpu_processor_backend.cpp:205`）；不支持/未实现的 YUV 转换在 `cvt_color` 入口显式拒绝（`image_data.cpp:474-489`），不再静默走错路径。
- 位置：`image_data.cpp:104-113`（`planes()` 只特判 NV12/NV21）、`:322-358`（`toCpu` 无 I420 分支）、`convert.cpp`（转换表有 I420 映射）
- 说明：枚举 + `from_raw` 接受 I420，但平面派生 / toCpu / codec 守卫（只认单平面）全不支持 → I420 帧静默走错路径。
- 备注：唯一消费方 capi `md_image_from_yuv420p`（`md_capi.cpp:368-375`）根本没走 ImageData，直接裸 `cv::Mat` + `cvtColor`。

### 11. `from_device_planes` 命名泄漏历史 + 签名写死 NV12

- **状态：已解决（2026-08-18 Task 7 复核）**：已更名/泛化为 `from_planes`（`image_data.h:75`），支持任意平面数（1–3）与 fmt。
- 位置：`image_data.cpp:242-268`
- 说明：机制本身设备无关（host NV12 一直在走它，`md_capi.cpp:321`、`test_image_data.cpp:395`），名字却叫 "device"；fmt 硬编码 NV12、固定 2 平面、ch=1、nbytes=1.5wh，无法表达 NV21 / I420。

### 12. `from_bgr24` 与 `from_raw(copy=false)` 重复

- **状态：受控（2026-08-18 Task 7 复核）**：`from_bgr24` 现为 `from_raw(...,PKG_BGR_U8,false)` 的薄别名（`image_data.cpp:312`），保留为便捷 API，重复 Storage 路径已消除。
- 位置：`image_data.cpp:270-295`
- 说明：同为"CPU 借用 BGR、不复制"，却产生两种不同 Storage、两条维护路径。C API `md_image_from_bgr24` 内部早已走 `from_raw`（`md_capi.cpp:285`），C++ 静态方法仅剩测试引用。

### 13. `const cv::Mat&` 构造 = 浅共享陷阱

- **状态：已解决（2026-08-18 Task 7 复核）**：`ImageData(const Mat&)` 现深度拷贝进自有 `owner` 缓冲（`image_data.cpp:151-173`），不再与源 `Mat` 共享底层数据。
- 位置：`image_data.h:28`
- 说明：`ImageData(m)` 与 `m` 共享底层数据，后续修改 `m` 直接可见。全仓库仅 2 处使用（`vis_sem.cpp:57`、`vis_depth.cpp:55`，均同作用域），价值低、陷阱大。

---

## P2 · 中等（过度设计 / 性能）

### 14. 每次 op 新建 backend

- **状态：仍成立（2026-08-18 Task 7 复核）**：`backend_for` 每次 `make_unique`；已由 `supports(device,op)` 前置 fast-fail 缓解"必然失败仍 acquisition"的浪费，但 TPU/CUDA 后端仍每次构造/析构。留待后续。
- 位置：`image_data.cpp:24-26`（`backend_for`）
- 说明：`crop`/`rotate`/`resize`/`cvt_color` 每个调用都 `make_unique` 一个新 backend。CPU 后端无状态无所谓，但 TPU 后端构造 = `bm_dev_request`（`sophgo_processor_backend.cpp:22-32`）、析构 = `bm_dev_free`（`:34-45`），而 Sophgo / CUDA 又把这三个 op 覆写为**直接 `return false`**（`sophgo_processor_backend.h:58-70`、`cuda_processor_backend.h:71-84`）——完整设备 acquire 一次，换来一次必然失败。

### 15. 7 个构造入口覆盖 3 类场景

- **状态：部分解决（2026-08-18 Task 7 复核）**：构造入口已收敛为 5 个构造 + `from_raw`/`from_planes`/`from_bgr24` 工厂；`from_device_planes` 已并入 `from_planes`，重叠消解。剩余 `from_bgr24` 与 `from_raw` 薄别名保持。
- 位置：`image_data.h:26-29,62-78`
- 说明：本质场景只有 3 类（CPU 分配 / CPU 借用 / 多平面借用），实际提供 7 个入口（default、尺寸构造、双 Mat 构造、from_raw、from_bgr24、from_device_planes），其中两两重叠（P1-9、P1-12）。

### 16. 元数据填写散落

- **状态：部分解决（2026-08-18 Task 7 复核）**：`make_owned` 已作为自有内存统一入口集中推导布局/字节数；`from_raw`/`from_planes` 简化为少数字段手工填。剩余 `toCpu`/clone 逐字段复制仍存在。
- 位置：`image_data.cpp:249-293`（三个 `from_*` 各手工填 8 字段）、`:309-367`（`toCpu` CPU 分支再逐字段复制 10 行）
- 说明：无单一"impl 克隆/刷新"入口，新增字段即静默失同步（`refresh_meta` 的注释已自认"PlaneStorage 宽高通道已是构造时填好的真实值，不清零"）。

---

## P3 · 轻微（死代码 / 命名 / 债）

### 17. `keeper` 字段承诺保活却从未赋值

- **状态：已解决（2026-08-18 Task 7 复核）**：`keeper` 字段已删除，改用统一 `owner`（`shared_ptr<void>`）字段承载借用源保活。
- 位置：`image_data.cpp:42`
- 说明：注释"借用源 RAII（可空），保活"，全仓库无一处设置。一个给出错误安全暗示的成员，比没有更危险（借用 API 的寿命契约因此含糊）。

### 18. 零调用成员

- **状态：部分解决（2026-08-18 Task 7 复核）**：`is_cpu_plane()` / `set_cpu_plane_dims()` 已随统一存储删除；`element_count`/`imshow` 等仍保留（有/近测试引用）。`is_shared_with` 仍零调用。
- 位置：`image_data.cpp:86-96`（`is_cpu_plane()` / `set_cpu_plane_dims()`）、`:226`（`is_shared_with`）、`image_data.h:49-50`（`element_count` / `element_bytes`，仅前者有 1 处测试引用）、`image_data.h:85`（`imshow`，唯一调用是注释行 `demo_detection_cxx.cpp:100`）

### 19. `format()` 是 `type()` 纯别名

- **状态：仍成立（2026-08-18 Task 7 复核）**：接受的命名冗余，保留。

### 20. 僵尸 benchmark 文件

- **状态：已解决（2026-08-18 Task 7）**：`benchmark/benchmark_image_data.cpp`、`benchmark/benchmark_yolo_preproc.cpp` 均已删除；`benchmark/CMakeLists.txt:8` 仅编 `benchmark_main.cpp` + `benchmark_models.cpp`，无引用残留。
- 位置：`benchmark/benchmark_image_data.cpp`、`benchmark/benchmark_yolo_preproc.cpp`
- 说明：仍在调用已删除的 `cast` / `pad` / `normalize` / `letter_box` / `center_crop` / `permute` / `convert` / `fuse_*` 方法，编译不过，且不在 `add_executable` 内（`benchmark/CMakeLists.txt:8` 只编 `benchmark_main.cpp` + `benchmark_models.cpp`）——误导性遗留。

### 21. `ImageData(w,h,NV12)` 静默空图

- **状态：已解决（2026-08-18 Task 7 复核）**：`make_owned`（`image_data.cpp:86-119`）现按 fmt 推导平面布局——NV12/NV21 建 2 平面、I420 建 3 平面、packed 建 1 平面，`ImageData(width,height,NV12)` 不再返回空图。
- 位置：`image_data.cpp:148-156`
- 说明：YUV 类型 `ocv_type == -1` → `cv::Mat(h, w, -1)` 为空，构造函数不报错不告警，返回空图。

### 22. 文档滞后

- **状态：已解决（2026-08-18 Task 7）**：`docs/preprocess.md:7` 已更新为"统一平面存储 + `cv::Mat` 桥"描述。
- 位置：`docs/preprocess.md:7`
- 说明：仍描述 ImageData"底层基于 OpenCV Mat"，与改造后的平面模型不符。

---

## 根因归纳（22 条：原 #5 已并入 #1；#1–4、#6–13、#17、#20–22 已在统一平面存储改造（Task 1–6）+ 清理（Task 7）中解决）

三条根因的解决情况：
1. **双表示** → 已解决（统一为单 `ImageDataImpl` 平面存储）
2. **Mat 锚点未拆除** → 已解决（`cv::Mat` 降为桥，操作全部经后端分派）
3. **迁移不彻底** → 大部落已解决（#11/#12/#17/#20/#21/#22）
   （剩余待办：#14、#15、#16、#18、#19）

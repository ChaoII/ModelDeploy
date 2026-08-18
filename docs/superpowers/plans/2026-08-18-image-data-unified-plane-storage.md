# ImageData 统一平面存储重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把 `ImageData` 重构为统一平面存储 `{fmt,w,h,ch,device,planes[3],owner}`，`cv::Mat` 仅作桥；构造族收敛 5 个；操作全走后端分派（设备帧 fast-fail 不建 backend）；YUV 可构造自描述、无转换操作显式报错；C API 图像操作一律经 ImageData 桥。

**Architecture:** 单一 `ImageDataImpl` 持有格式/宽高/通道/设备/3 平面/owner（保活）。删除 `CpuStorage`/`PlaneStorage` 双表示与 `mutable plane_mat_`/`refresh_meta`（元数据仅构造时填一次）。`asMat()` 仅对 packed 格式产 CPU 借用视图。所有图像操作（含 rotate_crop/PA2PL/PL2PA）收束到 `VisionProcessorBackend` 分派，`supports(device,op)` 前置让设备帧 fast-fail 而不建 backend。C API `md_image_*` 不再自造 `cv::Mat`，统一经 `asMat()`/`plane(i)` 桥。

**Tech Stack:** C++17（无异常，部分 TU 无 `/EHsc`，有 C4530）、OpenCV（仅作桥）、Catch2（单测）、pybind11。构建：`build_tdc`（WITH_GPU=OFF、SOPHGO=OFF、BUILD_TESTS=ON）；命令只能 `cmd /c '"...vcvars64.bat" >nul 2>&1 && cmake --build build_tdc [--target test_modeldeploy] --parallel'`，**不要重配 CMake**。测试在 `build_tdc\bin`；需模型/数据的用 `TEST_DATA_DIR=E:\CLionProjects\ModelDeploy`。

## Global Constraints

- **无异常**：部分 TU 无 `/EHsc`（C4530）；错误一律走 `ImageData::last_error`/`set_last_error`（thread_local），**不 throw**。
- **P0 全随统一存储结构消解或显式守卫**（登记册 #1/#2/#3/#4/#5/#6/#7/#9/#10/#14/#16/#21）。
- **对外 C ABI 签名不变**（`md_*` 不动）；`ImageData` 公开方法名 `plane(i)/asMat/toCpu/clone/cvt_color/...` 保留。
- **C API 图像操作一律走 ImageData 桥**：不自行 `cv::Mat(hi->height,hi->width,CV_8UC3,hi->data)`；经 `asMat()`（packed-only）或 `plane(i)`。
- **设备帧 fast-fail 不建 backend**：`supports(device,op)` 前置。
- **YUV 无转换反走显式报错**：仅 `PKG_BGR/RGB` 互转 + 专用 `CVT_NV122PKG_BGR` + 新增 `CVT_I4202PKG_BGR` 可转换。
- 迁移一律保持 CPU 行为逐位一致（回归锁定）。全量非模型 `~[model]` 保持 ≥ 1094/113 全绿。
- 基线 HEAD：`f800c03`。消费调研：`sdd/unified-storage-consumers.md`。登记册：`docs/image_data_issue_register.md`（基线 `f800c03`）。

---

### Task 1: 统一平面存储（核心）— `ImageDataImpl` + 5 构造 + `asMat` packed-only

**Files:**
- Modify: `csrc/vision/common/image_data.h`（整体重构声明区）
- Modify: `csrc/vision/common/image_data.cpp`（实现区）
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: 现状 `MdImageType`、`ColorConvertType`、`VisionProcessorBackend`、`Device`。
- Produces:
  - `struct ImageData::Plane { const uint8_t* data; int step; };`
  - `struct ImageDataImpl { MdImageType fmt; int w,h,ch; Device device; std::array<Plane,3> planes; std::shared_ptr<void> owner; bool cmat_dirty=false; };`（内部，放在 .h 的 ImageData 私有/或 .cpp 匿名域——用匿名域则 .h 只留 `std::shared_ptr<void> impl_`，其余方法在 .cpp）
  - 构造：`ImageData()` / `ImageData(int w,int h,MdImageType)` / `explicit ImageData(const cv::Mat&)`（吸收 `Mat&&`）/ `static from_raw(uint8_t*,int,int,MdImageType,bool copy=false,Device=CPU,std::shared_ptr<void>={})` / `static from_planes(const Plane*,size_t,MdImageType,int,int,Device={},std::shared_ptr<void>={})` / `static from_bgr24(const uint8_t*,int,int)`（别名）
  - `bool asMat(cv::Mat*) const`：packed-only
  - `int width()/height()/channels()` 读 impl 字段
  - 移除/改造后仍实现的：`plane_count()/plane(i)/empty()/clone()/toCpu()/crop()/rotate()/rotate_crop()/resize()/cvt_color()/imshow()/bytes()/element_count()/element_bytes()/is_shared_with()/type()/format()/device()`

**说明（实现要点）：** `.h` 中 `cv::Mat` 若要成员，用抽象 pimpl：`struct ImageDataImpl; std::shared_ptr<ImageDataImpl> impl_;`（`ImageDataImpl` 定义放 .cpp 匿名域，含 3 平面 + owner；不再含 `cv::Mat` 成员）。`asMat` 现用 `cv::Mat(h,ch,CV_8UC3,planes[0].data,planes[0].step)` 构造临时视图（CPU packed）。删除 `const Mat&`/`Mat&&` 双构造里的 storage 分支，统一为"摄取到平面：packed→`planes[0]={mat.data, mat.step}`；非 packed→置错空图"。删除 `materialize_plane_mat`/`plane_mat_`/`refresh_meta`/`is_cpu_plane`/`set_cpu_plane_dims`/`keeper` 老路径。`toCpu`/`clone` 在本 Task 只做"结构不动、逻辑从新存储走"，其守卫细化为 Task 4/5。

- [x] **Step 1: 写失败测试（构造/高度/自描述/asMat）**

在 `tests/test_image_data.cpp` 追加（先让当前实现失败，锁定新契约）：
```cpp
TEST_CASE("image_data unified storage: ctor/family/height/asMat", "[image_data]") {
    // (a) 自分配 NV12 不再空（height 恒真实 h）——旧实现 #21 空图
    ImageData nv(w=64, h=48, MdImageType::NV12);
    CHECK(!nv.empty());
    CHECK(nv.width() == 64); CHECK(nv.height() == 48);      // 真实 h，非 1.5h
    CHECK(nv.plane_count() == 2);
    // (b) from_raw(NV12) 平面自描述、height==h —— 旧实现在此给 1.5h
    std::vector<uint8_t> buf(64*48*3/2); std::fill(buf.begin(),buf.end(),100);
    auto fr = ImageData::from_raw(buf.data(), 64, 48, MdImageType::NV12, true);
    REQUIRE(!fr.empty()); CHECK(fr.height()==48); CHECK(fr.plane_count()==2);
    CHECK(fr.plane(0).step == 64); CHECK(fr.plane(1).data == fr.plane(0).data + 64*48);
    // (c) asMat packed-only
    auto bgr = ImageData::from_raw(nullptr_or_buf /*PKG_BGR 4 像素*/, 2,2, MdImageType::PKG_BGR_U8, false);
    cv::Mat m; CHECK(bgr.asMat(&m)); CHECK(m.rows==2 && m.cols==2 && m.channels()==3);
    cv::Mat mn; CHECK(!nv.asMat(&mn));   // YUV → asMat=false
    // (d) from_bgr24 别名行为一致
    auto fb = ImageData::from_bgr24(/*bgr*/, 2,2);
    CHECK(fb.format()==MdImageType::PKG_BGR_U8); CHECK(fb.plane(0).data != nullptr);
}
```
（测试内具体 buffer 用真实字节数组，此处示意。）

- [x] **Step 2: 运行确认失败**

Run（cwd `build_tdc\bin`，`TEST_DATA_DIR` 设仓库根）: `.\test_modeldeploy.exe "image_data unified storage: ctor/family/height/asMat"`
预期：`#21`（自分配 YUV 空）、`height==1.5h` 或 `asMat` YUV 未拒 等 FAIL。

- [x] **Step 3: 实现统一存储**

`image_data.h`：把 `impl_` 改为 pimpl（`struct ImageDataImpl; std::shared_ptr<ImageDataImpl> impl_;`），声明新构造签名（见 Interfaces），保留 `Plane`。
`image_data.cpp`：
```cpp
namespace { struct ImageDataImpl {
    MdImageType fmt = MdImageType::PKG_BGR_U8;
    int w=0,h=0,ch=1; Device device = Device::CPU;
    std::array<ImageData::Plane,3> planes{}; size_t nplanes=0;
    std::shared_ptr<void> owner;
    size_t bytes_=0, element_count_=0, element_bytes_=1;
    bool cmat_valid=false; cv::Mat cmat_;  // asMat 临时物化缓存（packed），非共享数据源
}; }
// 数据源唯一：planes[]；bytes/element 由构造时按 fmt 计算。
```
- 元数据单点填入：helper `finalize(img,w,h,fmt,device,owner)` 内一次性算 `bytes_/element_count_`。
- 构造器（示意）：
```cpp
ImageData::ImageData(int w,int h,MdImageType t){ impl_=std::make_shared<ImageDataImpl>();
  impl_->fmt=t; impl_->w=w; impl_->h=h; impl_->device=Device::CPU;
  const int ocv = md_image_type_to_ocv_type(t);
  int ch = (ocv>0)? CV_MAT_CN(ocv) : (is_planar(t)? 1 : 0);
  auto buf=std::make_shared<std::vector<uint8_t>>(md_image_type_bytes(t,w,h));
  uint8_t* p0=buf->data(); impl_->owner=buf;
  if (t==NV12||t==NV21){ impl_->planes[0]={p0,w}; impl_->planes[1]={p0+(size_t)w*h,w}; impl_->nplanes=2; impl_->ch=1;}
  else if (t==I420){ impl_->planes[0]={p0,w}; impl_->planes[1]={p0+(size_t)w*h,w/2}; impl_->planes[2]={p0+(size_t)w*h*5/4,w/2}; impl_->nplanes=3; impl_->ch=1;}
  else { impl_->planes[0]={p0, (int)(w*ch)}; impl_->nplanes=1; impl_->ch=ch; impl_->bytes_=w*h*ch; }
  impl_->element_count_=w*h; impl_->element_bytes_=ch; }
```
- `from_raw`：`data` 需 `const_cast`；`copy`→自分配 buffer 并 `memcpy`（沿用上面布局）；`!copy`→`planes[]` 指向 `data`，`owner=owner`（借）。`device` 透传。
- `from_planes(planes,n,fmt,w,h,device,owner)`：直接搬 planes、`nplanes=n`、`ch`/`bytes` 按 fmt 推。
- `from_bgr24(bgr,w,h)` = `return from_raw(const_cast<uint8_t*>(bgr),w,h,PkgBGR_U8,false,CPU);`（内联在 .cpp）。
- `asMat(out)`：仅当 `fmt` 为 packed（`PKG_BGR_U8/RGB_U8/BGRA/RGBA`）且 `device==CPU`：`*out = cv::Mat(h,w,CV_MAKETYPE(CV_8U,ch), const_cast<uint8_t*>(planes[0].data), planes[0].step); return true;` 否则 `set_last_error("asMat: packed CPU only")` + `return false`。
- 其余老化访问 `impl_->mat()` 的成员（rotate_crop/imshow/clone/toCpu/cvt_color PA2PL）本 Task **临时**改为"用 asMat+plane 重写其取图入口"或先返回空+last_error（Task 4 正式落 CPU backend）。`toCpu`/`clone` 先按 planes 直接搬运（见 Task 4/5 守卫）。
- `width()/height()/channels()/device()/type()/format()/plane_count()/plane(i)/empty()/bytes()/element_count()/element_bytes()/is_shared_with()` 直接读 impl 字段。

- [x] **Step 4: 运行测试通过**

Run: `.\test_modeldeploy.exe "image_data unified storage*"` 预期 PASS；随后 `.\test_modeldeploy.exe "~[model]"` 先接受因迁移未完成导致的"编译错/局部失败"（本计划各 Task 逐步还原），以 Task 8 最终全绿为准。

- [x] **Step 5: Commit**

```bash
git add csrc/vision/common/image_data.h csrc/vision/common/image_data.cpp tests/test_image_data.cpp
git commit -m "feat(image): unified plane storage {fmt,w,h,ch,device,planes,owner}; 5 ctors; asMat packed-only"
```

---

### Task 2: `supports()` 分派前置 + `from_raw` 携带 device/owner

**Files:**
- Modify: `csrc/vision/processors/processor_backend.h`（+ `enum class ImageOp`，+ `static bool supports(Device, ImageOp)`）
- Modify: `csrc/vision/processors/cpu/cpu_processor_backend.cpp`（如已内联重写处保持）
- Modify: `csrc/vision/common/image_data.cpp`（`crop/rotate/resize/cvt_color` 入口前置检查）
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: Task 1 新存储 + `Device`。
- Produces: `ImageOp{Preprocess,Draw,Crop,Rotate,Resize,CvtColor,RotateCrop,PlaneSplit}`；`VisionProcessorBackend::supports(Device,ImageOp)` 返回 bool；`ImageData` op 入口统一：
```cpp
if (!VisionProcessorBackend::supports(device(), ImageOp::Resize)) { set_last_error("resize: device unsupported"); return ImageData(); }
auto& bk = *backend_for(device());  // 至此才建 backend
```

- [x] **Step 1: 写失败测试（设备帧 op fast-fail 不建 backend）**

构造一个 `from_planes`(device=GPU, NV12, 两个哑指针 y/uv) 的帧，然后对同一帧依次 `crop/rotate/resize/cvt_color/rotate_crop`，断言**全部返回空 + 设置了 last_error**（不再走 backend，不崩溃）。占位测试先对旧实现行为断言（当前这些会建 backend 且 CUDA 桩返回 false——行为仍空，但**无 fast-fail 语义**；测试用 `last_error` 文案断言区分）。旧实现 `supports` 不存在 → 编译失败。

- [x] **Step 2: 运行确认失败（编译缺符号）**

Run: `cmake --build build_tdc --target test_modeldeploy` 预期：找不到 `VisionProcessorBackend::supports`。

- [x] **Step 3: 实现**

`processor_backend.h`：
```cpp
enum class ImageOp { Preprocess, Draw, Crop, Rotate, Resize, CvtColor, RotateCrop, PlaneSplit };
class VisionProcessorBackend {
 public:
  static bool supports(Device d, ImageOp op) {
    if (d == Device::CPU) return true;
    return op == ImageOp::Preprocess || op == ImageOp::Draw;   // GPU/TPU 仅此两类
  }
  ...
};
```
`image_data.cpp` 各 op 入口（crop/rotate/resize/cvt_color/rotate_crop）统一前置：
```cpp
ImageData ImageData::resize(int width,int height) const {
  if (!impl_||impl_->empty()||width<=0||height<=0) return ImageData();
  g_last_error_msg.clear();
  if (!VisionProcessorBackend::supports(device(), ImageOp::Resize)) { set_last_error("resize: device unsupported (fast-fail)"); return ImageData(); }
  auto& bk = *backend_for(device());
  ImageData out; if (bk.resize(*this,&out,width,height)) return out;
  set_last_error("resize: backend failed"); return ImageData();
}
```
同法改造 `crop/rotate`。`cvt_color` 与 `rotate_crop` 在 Task 4 落 backend；本 Task 先给 cvt_color/rotate_crop 加同样 `supports` 前置（设备帧先拦住），其 CPU 实现见 Task 4。

- [x] **Step 4: 运行测试通过**

Run: `.\test_modeldeploy.exe "image_data*"` 及新 fast-fail 用例 PASS；`~[model]` 逐步恢复。

- [x] **Step 5: Commit**

```bash
git add csrc/vision/processors/processor_backend.h csrc/vision/common/image_data.cpp tests/test_image_data.cpp
git commit -m "feat(image): supports(device,op) gate so device frames fast-fail without building backend"
```

---

### Task 3: `from_planes` 泛化 + 迁移 `from_device_planes` + capi NV12/I420 接线 + `CVT_I4202PKG_BGR`

**Files:**
- Modify: `csrc/vision/common/image_data.h`（+`from_planes` 声明，`-from_device_planes` 或标 deprecated）
- Modify: `csrc/vision/common/image_data.cpp`（`from_planes` 实现；+ 内部 `device_supports_cvt`）
- Modify: `csrc/vision/common/basic_types.h`（`+CVT_I4202PKG_BGR`）
- Modify: `csrc/vision/common/convert.cpp`（枚举↔ocv 映射补 `CVT_I4202PKG_BGR→COLOR_YUV2BGR_I420`）
- Modify: `csrc/vision/processors/cpu/cpu_processor_backend.cpp`（`cvt_color` 加 `CVT_I4202PKG_BGR` 分支）
- Modify: `capi2/md_capi.cpp`（`md_image_from_nv12/device_nv12/yuv420p` 走 `from_planes`/`cvt_color`）
- Test: `tests/test_image_data.cpp`、`tests/test_capi.cpp`

**Interfaces:**
- Consumes: Task1 `from_planes`、`supports`。
- Produces: `ImageData::from_planes(const Plane*, size_t n, MdImageType fmt, int w,int h, Device={}, std::shared_ptr<void>={})`；`ColorConvertType::CVT_I4202PKG_BGR`；capi `md_image_from_nv12/device_nv12` 内部改 from_planes；`md_image_from_yuv420p` = `from_planes(I420)` + `cvt_color(CVT_I4202PKG_BGR)`。

- [x] **Step 1: 迁移调用点（13 处 from_device_planes）**

按 `sdd/unified-storage-consumers.md` 所列逐处改 `from_device_planes(y,uv,w,h,step_y,step_uv,device)` → `from_planes`：
```cpp
// GPU/CUDA 帧（device=GPU/TPU, 2 平面）
ImageData::Plane pl[2] = {{y, step_y>0?step_y:w},{uv, step_uv>0?step_uv:w}};
auto img = ImageData::from_planes(pl,2,MdImageType::NV12, w,h, dev, /*owner*/{});
// host NV12 借用（device=CPU）同式，owner 传保活（若有）。
```
涉及：`capi2/md_capi.cpp:321`（from_nv12）、`md_image_from_device_nv12`、后端 letterbox（cpu/cuda/sophgo 的 `md_ln...`/NV12 预处理调用处，若它们用 from_device_planes 则改）、`test_image_data.cpp:395/560-589` 等。删除 `from_device_planes` 声明（或保留标记 deprecated，按审阅定——默认删除，全部迁移后无引用）。

- [x] **Step 2: 新增 CVT_I4202PKG_BGR**

`basic_types.h` 枚举 `CVT_NV122PKG_BGR,` 后加 `CVT_I4202PKG_BGR`；`convert.cpp` ocv 映射 `case CVT_I4202PKG_BGR: return cv::COLOR_YUV2BGR_I420;`；`cpu_processor_backend.cpp::cvt_color` 加分支：
```cpp
case ColorConvertType::CVT_I4202PKG_BGR: {
  const auto p0=image.plane(0), p1=image.plane(1), p2=image.plane(2);
  if (!p0.data||!p1.data||!p2.data||image.plane_count()<3) return false;
  const int w=image.width(), h=image.height();
  std::vector<uint8_t> flat((size_t)w*h*3/2);
  // 三平面并入 flat（Y[h*w], U[h/2*w/2], V[...]）
  ... memcpy Y/U/V ...
  cv::Mat f(h*3/2, w, CV_8UC1, flat.data());
  cv::Mat bgr; cv::cvtColor(f, bgr, cv::COLOR_YUV2BGR_I420);
  cv::Mat bgr_buf; if (!bgr.isContinuous()) bgr=bgr.clone();
  *out = ImageData(bgr_buf);   // 或 from_raw 拷贝
  return true; }
```

- [x] **Step 3: capi 接线**

`md_image_from_nv12`（:354-375）与 `md_image_from_device_nv12`（:376-403）内部改 `from_planes(NV12,...)`；`md_image_from_yuv420p`（:406-413）改为：
```cpp
ImageData::Plane pl[3] = {{y, w},{u, w/2},{v, w/2}};  // 由平铺 data 拆三平面（或直接单 buffer 平铺再走 from_raw）
auto nv = ImageData::from_planes(pl,3,MdImageType::I420, w,h, /*CPU*/);
auto bgr = ImageData::cvt_color(nv, ColorConvertType::CVT_I4202PKG_BGR);
if (bgr.empty()) return MD_ERR_IMAGE_DECODE;  // 内部 set_error 由 cvt 填
```
（若平铺单 buffer，用 `from_raw(data,w,h,I420,true?false)` 亦可；选 from_planes 自解析一致。）

- [x] **Step 4: 测试（像素级等价）**

`test_capi.cpp`：`md_image_from_yuv420p` 与参考 `cvtColor(COLOR_YUV2BGR_I420)` 逐像素一致（仿现有 NV12 等价测试）；NV12 等价测试保持全绿（CVT_NV122PKG_BGR 不动）。`test_image_data.cpp`：`from_planes(NV12/I420/NV21)` 自描述（plane_count/step/height）断言。

- [x] **Step 5: 运行 + Commit**

Run: `.\\test_modeldeploy.exe "capi2*"` 与 `"*nv12*"` 全绿；`commit -m "feat(image): from_planes generalized; CVT_I4202PKG_BGR; capi nv12/yuv420p rerouted through ImageData"`

---

### Task 4: rotate_crop/PA2PL/PL2PA 入 CPU backend + imshow/clone 守卫

**Files:**
- Modify: `csrc/vision/processors/processor_backend.h`（虚方法 +rotate_crop）
- Modify: `csrc/vision/processors/cpu/cpu_processor_backend.{h,cpp}`（实现）
- Modify: `csrc/vision/common/image_data.cpp`（`rotate_crop` 走后端；`imshow/clone` 守卫）
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: Task1 桥（asMat/plane）、Task2 supports。
- Produces: `VisionProcessorBackend::rotate_crop(const ImageData&, std::array<float,8>, ImageData*)` 虚函数；CPU 实现 = 现 `image_data.cpp:433-487` 逻辑但基于 `asMat`；`ImageData::rotate_crop` → supports 前置 + `bk.rotate_crop`。

- [x] **Step 1: 迁移 rotate_crop**

把 `image_data.cpp::rotate_crop` 的纯 mat 逻辑搬到 `cpu_processor_backend.cpp::rotate_crop(const ImageData& img, std::array<float,8> box, ImageData* out)`（`img.asMat(&src)` 取 packed 视图后跑原透视逻辑，`*out = ImageData(dst)`）。`image_data.cpp::rotate_crop` 改为 supports 前置 + 调 backend（同 Task2 模式）。搬移后 `ppocr.cpp:119` 用法不变（接口同签名）。

- [x] **Step 2: migrate PA2PL/PL2PA**

`image_data.cpp::cvt_color` 的 `CVT_PA_BGR2PL_BGR/RGB2PL_RGB`（:518-575）搬进 `cpu_processor_backend.cpp::cvt_color` 分支（用 `asMat`/`plane(0)` 做 `cv::split`/`merge`），`image_data.cpp` 中删该裸分支，统一走 `bk.cvt_color`（PA2PL 的"device 拒绝"由 supports(CvtColor) 前置承担，不再需要 :521 手写守卫）。

- [x] **Step 3: imshow / clone 守卫**

`imshow`：`supports-device` 语义上显示仅 CPU——`if (device()!=CPU){ set_last_error("imshow: CPU only"); return; }` 后 `asMat(&m)` → `cv::imshow`。
`clone`：CPU 帧→按平面深拷贝到新 owner（复用构造布局）；非 CPU→`set_last_error("clone: device frame not supported")` + 返回空（修 #3）。

- [x] **Step 4: 测试**

`tests/test_image_data.cpp`：`rotate_crop` 与逐位参考（现行为）一致；设备帧 `rotate_crop`/`imshow` 返回空+last_error；`clone` 设备帧报错、CPU 帧深拷贝（改副本不影原）。

- [x] **Step 5: Run + Commit**

`commit -m "refactor(image): rotate_crop/PA2PL/PL2PA into CPU backend; imshow/clone add device guards"`

---

### Task 5: YUV 操作显式拒绝 + CPU-only NV12→BGR 补测

**Files:**
- Modify: `csrc/vision/common/image_data.cpp`（`toCpu`/`cvt_color`/codec 守卫）
- Modify: `csrc/vision/common/image_data.h`
- Test: `tests/test_image_data.cpp`

**Interfaces:**
- Consumes: Task1 存储、Task2 supports。
- Produces: `toCpu` 对 NV21/I420 → false + `set_last_error("toCpu: unsupported YUV <fmt>")`；`cvt_color` 对无转换格式（NV21/NV21 无专用枚举）→ false+报错；`codec_supported`/`imencode/imwrite` 对多平面 YUV → false+报错。

- [x] **Step 1: toCpu / cvt_color / codec 拒绝**

`toCpu`：保留 NV12→CPU 路径（现 :325-343 平面搬运）；新增前置 `if (fmt==NV21||fmt==I420){ set_last_error(...); return false; }`。
`cvt_color`：`supports(CvtColor)` 已拦设备；对 CPU 帧，在 `ocv_type` 映射后，若 `type` 为 `CVT_NV21*`/`CVT_I4202PA_*`（除新增 I4202PKG_BGR）而无 CPU 正确实现 → 显式 `set_last_error("cvt_color: unsupported YUV conversion")` + 空。
`imencode/imwrite/save`：沿用 `codec_supported`（非单平面拒绝），补 NV21/I420 明确 `last_error` 文案。

- [x] **Step 2: CPU-only NV12→BGR 补测（修 #5）**

`tests/test_image_data.cpp` 增：`from_raw(NV12, copy=true)` → `cvt_color(CVT_NV122PKG_BGR)` 结果与 `cvtColor(COLOR_YUV2BGR_NV12)` 逐像素等价（覆盖此前零单测的路径）。

- [x] **Step 3: Run + Commit**

`commit -m "feat(image): YUV ops (toCpu/cvt_color/codec) explicitly reject NV21/I420; add CPU-only NV12->BGR test"`

---

### Task 6: C API 图像操作全部走 ImageData 桥（§6）

**Files:**
- Modify: `capi2/md_capi.cpp`
- Test: `tests/test_capi.cpp`

**Interfaces:**
- Consumes: `asMat()`(packed-only)、`plane(i)`、`cvt_color`。
- Produces: `md_image_from_rgb24`/`show`/`save`/`encode`/`draw_rect/line/circle` 内部经 `ImageData` 桥。

- [x] **Step 1: 逐函数改写（不再自造 cv::Mat）**

- `md_image_from_rgb24`（:345）：`from_raw(rgb,PKG_RGB_U8,false)` → `cvt_color(CVT_PA_RGB2PA_BGR)` → 装 handle。
- `md_image_show`（:475）：`hi->image.asMat(&m)`；`if(!asMat) return MD_ERR_UNSUPPORTED_TYPE;`（YUV/设备→报错）→ `cv::imshow`。
- `md_image_save`（:490）/`md_image_encode`（:499）：先 `asMat(&m)`，非 packed→`MD_ERR_UNSUPPORTED_TYPE`，再 `cv::imwrite`/`cv::imencode`。
- `md_draw_rect/line/circle`（:2001/2015/2030）：`hi->image.asMat(&m)`（CPU packed）→ 在 `m` 上绘制（不改句柄结构）；非 packed→报错（NV12 绘制仍走 `md_draw_result` device 路径）。
- 确认 `handle_has_cpu_bgr`（:190）路径最终都落 to asMat 语义。

- [x] **Step 2: 测试**

`tests/test_capi.cpp`：RGB24 进→出 BGR 手柄与 bgr24 进一致（逐像素）；设备 NV12 帧 `md_image_show/save/encode/draw_rect` → `MD_ERR_UNSUPPORTED_TYPE`；CPU BGR 帧 `encode/save` 仍 OK。

- [x] **Step 3: Run + Commit**

Run: `.\\test_modeldeploy.exe "capi2*"` 全绿；`commit -m "refactor(capi): image ops all route through ImageData bridge (asMat/plane/cvt_color)"`

---

### Task 7: 消费方对齐 + 清理债

**Files:**
- Modify: `python/`（pybind `Mat&&` 构造点 → `ImageData(const Mat&)` 吸收）、`csrc/utils/utils.cpp:409`、`csrc/vision/face/face_align/face_align.cpp:148`、`application/batch_scheduler.cpp:154`、`application/infer_group.cpp:210`、`benchmark/benchmark_yolo_preproc.cpp:54`
- Cleanup: `benchmark/benchmark_image_data.cpp`、`benchmark/benchmark_yolo_preproc.cpp`（僵尸，删除或修；确认不在 add_executable → 删除文件）——登记册 #20
- Modify: `docs/preprocess.md`（#22 描述改"平面存储 + mat 桥"）

**Interfaces:**
- Consumes: 新构造/`plane(i)`/`asMat`。
- Produces: 全仓库编译一致、无裸 mat 残留。

- [x] **Step 1: 消费点逐个对齐**

按 `sdd/unified-storage-consumers.md` 将剩余 `from_raw(NV12)` 生产调用（batch_scheduler/infer_group/benchmark）改为新平面语义读 `plane(0/1)`；pybind/~45 `Mat&&` 构造改 `const Mat&`；`utils/face_align` 构造点改新签名。
- [x] **Step 2: 清理僵尸**：删除 `benchmark/benchmark_image_data.cpp`、`benchmark/benchmark_yolo_preproc.cpp`（登记册 #20），若 main 仍引用则同步移除。`keeper` 若新存储已用 owner 则删旧字段（#17）。
- [x] **Step 3: 文档**：`docs/preprocess.md:7` 改描述；`docs/image_data_issue_register.md` 更新已消解项状态 + 基线。
- [x] **Step 4: Build + 回归**

Run: `cmake --build build_tdc --target test_modeldeploy` 全绿；`.\test_modeldeploy.exe "~[model]"` ≥1094/113 全绿。
- [x] **Step 5: Commit** `chore(image): align consumers to unified storage; remove zombie benchmark; update docs`

---

### Task 8: 全量回归 + 像素级验收

**Files:** 测试修正（如有）
**Interfaces:** 无新接口，验证性。

- [x] **Step 1: 回归基线**

Run: `.\test_modeldeploy.exe "~[model]"`（TEST_DATA_DIR 设仓库根）& `"capi2*"` & `"*nv12*"` & `"image_data*"` → 全绿。
- [x] **Step 2: 像素级等价抽查**：NV12→BGR、I420→BGR（yuv420p）、RGB24→BGR、`from_bgr24` 与 `md_image_from_bgr24` 逐位一致。
- [x] **Step 3: 设备帧 fast-fail 抽查**：GPU/TPU 帧 `crop/resize/rotate/cvt_color/rotate_crop` + capi `show/save/encode/draw_*` → 均报错不崩不静默。
- [x] **Step 4: Commit**（如有修正）
- [x] **Step 5: 登记册/ spec 状态收尾**：勾 `docs/superpowers/plans/2026-08-18-image-data-unified-plane-storage.md` 全部 checkbox；更新 `docs/image_data_issue_register.md` 各消解项状态。

---

## Self-Review 自检（作者对 spec 覆盖率）

- §1 统一存储 → Task 1 ✅
- §2 构造 5 个 + from_bgr24 别名 → Task 1 ✅
- §3 操作分派（supports fast-fail）→ Task 2 ✅；rotate_crop/PA2PL/PL2PA 入 backend → Task 4 ✅
- §4 YUV 支持/拒绝 → Task 3（构造+CVT_I4202PKG_BGR）+ Task 5（toCpu/cvt/codec 拒绝+补测）✅
- §5 P0 矩阵：#1/#2/#6/#7/#9/#14/#16/#21 → Task 1/2；#3/#4 → Task 4；#5 → Task 5；#10 → Task 3/5 ✅
- §6 C API 收敛 → Task 6 ✅
- §7 迁移/兼容 + §8 实施 → Task 3（capi NV12/I420）+ Task 7（pybind/app/benchmark/tests）+ Task 8（回归/文档）✅
- #15/#17/#18/#19/#20/#22 清理债 → Task 7 ✅

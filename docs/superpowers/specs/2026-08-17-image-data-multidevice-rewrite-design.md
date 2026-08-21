# ImageData 多设备统一重构 — 设计 Spec

日期：2026-08-17
分支：`capi-v2`
状态：设计已获批准（第 1~4 节逐节确认）

## 背景与问题

`ImageData` 当前是"cv::Mat + 后补 planes"的混合结构，存在三处设计缺陷（均有代码证据）：

1. **设备帧不自描述**：`ImageDataImpl::refresh_meta()` 在 `device != CPU` 时把 `width/height/channels` 全部清零
   （`image_data.cpp`）。设备帧因此连自身宽高都没有，capi 不得不在外面另存 `fh->width/height`
   （`md_capi.cpp`），容器描述不了自己，元数据外挂。
2. **强依赖 OpenCV**：`cv::Mat` 是 CPU 模式的唯一真相锚点，planes 是后加的补丁；公开 API
   （`data()/y()/uv()/to_mat()`）围绕 cv::Mat 展开。
3. **capi 越俎代庖**：`md_image_from_nv12`（CPU cvtColor 拷贝）、`md_image_crop`、`md_draw_rect`
   等在 `md_capi.cpp` 里各自用 cv::Mat 复刻 CPU 逻辑，与 C++ 层措辞脱节、设备盲；而
   `md_draw_result` 又已做过按 device 分派（draw_gpu / draw_bmcv）。同一套逻辑两处实现。

另：模型级 `predict_nv12(...)` 是"ImageViewData 不足格"的临时补丁，除推理外还额外返回设备可视化帧
`out_frame`，本质应能被"自描述设备帧 + predict(ImageData) + 设备绘制"吸收。

## 目标

把 `ImageData` 重写为**多设备统一的容器**，以机械设备帧为第一目标，消除以上缺陷，并让
`predict(ImageData)` 成为唯一推理入口。

- 多设备统一作为第一目标（用户选档 C：彻底重写）。
- 允许改公开 API、接受大范围联动（用户选档 B）。
- 首版必须设备化的五大能力（用户多选全量）：
  1. 容器平面化（planes/device/元数据）
  2. 预处理器四件套（resize/crop/rotate/cvtColor）
  3. 绘制（CPU / CUDA / BMCV）
  4. 编解码（imread/imwrite/ encode/imdecode）
  5. 收敛 `predict_nv12` / `from_nv12`

## 方案（已定）：统一设备帧容器 + 操作分派

（其余备选：Frame/ImageData 分离两类型、模板特化——均否决，理由见 brainstorm 记录。）

## 第 1 节 · 容器与 Storage 架构

- 公开类型保留 `ImageData`（值类型，`shared_ptr<ImageDataImpl>`），重写内部。

  ```
  ImageData (值类型)
   └─ shared_ptr<ImageDataImpl>
       ├─ Device device            // CPU / GPU / TPU（OPENCL/VULKAN 预留）
       ├─ MdImageType format       // BGR / NV12 / RGB / GRAY / ...
       ├─ int width, height        // 设备帧也真实填写，不再清零
       ├─ int planes_count
       ├─ Plane planes[≤3]         // { uint8_t* data; int step; }
       ├─ Storage* storage         // 底层缓冲持有者（后端隔离）
       └─ ownership 标记
  ```

- `Storage` 承载抽象（内部，不暴露具体后端于公开头）：
  - `CpuStorage` — 包 `cv::Mat`（自有）或借用视图；CPU 既有逻辑照常走它。
  - `CudaStorage` — 包 CUDA buffer + pitch + 是否设备指针；零拷贝借用 GPU NV12 帧。
  - `TpuStorage` — 包 `bm_image` / TPU 显存句柄；零拷贝借用 TPU 帧。
  - 各提供 `device()`、平面访问、可选"复制到 CPU"能力。

- 公开 API 重塑要点（允许改签名）：
  - 保留 `width()/height()/channels()/format()/device()/empty()`。
  - 用 `plane(i) -> {data, step}` 统一取平面；`data()` 仅对单平面 CPU 便捷保留。
  - 去掉语义含糊的 `y()/uv()`；设备帧取 CPU 数据走显式 `toCpu()`。
  - `cv::Mat` 降级为 CPU Storage 的一种承载：只有 CPU 图能 `asMat()/toCpu()`，设备帧调用明确报 "not CPU"。

## 第 2 节 · 操作分派层 + predict(ImageData) 收敛

- `resize/crop/rotate/cvtColor/绘制/codec` 作为 `ImageData` 方法/静态函数，按 `device()`
  经 `VisionProcessorBackend`（CPU/CUDA/SophgoBMCV）分派；该机制从"局部实现"扶正为统一做法。
- 预处理器四件套：设备内优先（CUDA 走 cv-cuda/npp，BMCV 走 bmcv），零拷贝；后端未有该 op →
  明确报错，**绝不静默拷回 CPU**。cvtColor 同时承担格式转换（NV12↔BGR），设备感知。
- 绘制：走同一分派（CPU=OpenCV，CUDA=draw_gpu，BMCV=bmcv 已登记绘制），capi `md_draw_*`
  委托给这些 ImageData 方法，删掉 cv::Mat 复刻。
- 编解码：保持在 CPU（OpenCV），统一走 ImageData 方法；设备帧执行 encode/save → 明确 "not CPU"
  错误，不静默拷贝；需保存设备帧时显式 `toCpu()`。
- `predict(const ImageData&)` 成为唯一入口：识别 device≠CPU 帧，预处理器按 device 分派、设备内零拷贝。
- 模型级 `predict_nv12(...)` 移除；"进入设备帧"统一为 `from_device_planes(y,uv,w,h,step_y,step_uv,device)`
  （零拷贝包装，正确填宽高）。
- capi：删除 `md_model_predict_nv12`；新增 `md_image_from_device_nv12`（包设备帧）+ `md_model_predict`
  走通。`md_image_from_nv12`（CPU cvtColor 便捷）保留但重写为走 ImageData cvtColor。capi 其余图像
  方法改为薄委托给 ImageData。

## 第 3 节 · 错误处理与所有权/生命周期

- **不使用异常**（部分 TU 未开 `/EHsc`，构建有 `C4530` 警告）。沿用仓库风格：
  - 线程局部错误通道（对齐 capi `set_error/get_last_error`），操作失败写可读消息。
  - 就地操作（rotate、绘制、resize 就地）返回 `bool`；失败 `false` + last-error。
  - 返回值操作（crop/cvtColor/toCpu/imread/imwrite/encode）返回结果，失败返回"空"+ last-error。
  - 设备未实现 op → 明确 "device not supported" 错误，不静默降级。
- 所有权/生命周期：
  - `ImageData` 值类型 + `shared_ptr<Impl>`（浅拷贝不变）。
  - 设备帧默认借用外部缓冲（不拥有）；SDK 自分配缓冲（predict 输出帧/采集帧）经 `Storage` 挂
    `SharedBuf` RAII 保证存活期。
  - `clone()` 深拷贝到同设备。

## 第 4 节 · 测试与迁移

- 测试：
  - 既有 CPU 测试全部保持绿色（CPU 走 CpuStorage，行为等价）。
  - 新增：设备帧自描述（宽高非零）、`from_device_planes` 零拷贝、`predict(ImageData)` 设备 NV12
    帧结果 == 旧 `predict_nv12`、op 分派（CPU 必有；CUDA/BMCV 挂设备标签）、错误通道（设备未实现 op 报错）。
  - 断言 `predict_nv12` 从公开 API 移除。
- 迁移序列（每阶段绿再进下一阶段）：
  1. 容器 + Storage + CPU Storage（修元数据 bug）
  2. 操作分派层 + 三 backend 补齐预处/绘制/格式接口
  3. 编解码统一走 ImageData
  4. 收敛 `predict(ImageData)`（删模型级 `predict_nv12`）+ `from_device_planes`
  5. capi 图像/绘制/预测改为薄委托
  6. pybind/绑定同步
  7. 删遗留 API + 全部回归 + 文档
- 1、2 为核心；3-6 联动；7 收尾。1-2 阶段同步提供 `asMat()/toCpu()` 迁移出口，避免 C++ 消费方一步全断。

## 非目标（YAGNI）
- 不新增第 4/5 设备（OPENCL/VULKAN 仅枚举预留）。
- 不把 MNN 列入首版设备绘制/预处分派（MNN 视为 CPU 系后端）。
- 不做一次性大改：严格遵守第 4 节迁移序列，每阶段独立可验证。

## 验收口径
- 全量 CPU 回归保持基础断言数不下降。
- 新增设备帧/零拷贝/predict 收敛测试通过。
- capi 图像、绘制、预测全部委托到 C++ ImageData（无 cv::Mat 复刻残留）。
- `predict_nv12` 公开 API 移除且无哨代码残留。

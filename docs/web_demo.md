# Web 模型演示（demo_server v2 · 懒加载）

面向浏览器的模型族演示：一个可执行 `demo_server` 同时提供 `ServingServer` REST 推理与静态
Web 页面，网页里就能上传图片试跑各类模型（检测 / 分类 / 实例分割 / 姿态 / 旋转框 / 语义分割 /
深度 / OCR / 人脸 / 车牌），并把推理结果可视化出来。

v2 采用**目录驱动 + 单槽懒加载**：启动只注册模型目录（不实例化任何模型），前端选中模型时才
异步加载，单槽语义保证同一时刻最多一个模型处于就绪、切换模型会自动卸载上一个。

## 用途

- **快速体验**：不用写客户端代码，浏览器即可验证各模型族的输入输出与可视化效果。
- **模型验收**：在真实权重下按下方"每族端到端验证"逐族确认加载 `ready`、推理非空。
- **教学 / 演示**：一张图展示 SDK 的模型覆盖与 ServingServer 能力。

## 架构

```
浏览器 (web_demo/index.html)
   │  GET  /v1/models                 (目录：id/type/labels/input_size/status)
   │  POST /v1/models/{id}/load       懒加载 → loading/ready/failed
   │  POST /v1/models/{id}/infer      {image: base64} → 逐族结果 JSON
   │  POST /v1/models/{id}/unload     卸载（单槽切换时自动发生）
   ▼
ServingServer (csrc/serving/server.cpp)
   ├─ 静态托管：GET / → application/web_demo/（构建时拷到 build/bin/web_demo）
   ├─ 目录：    ModelRepo 读手写 manifest（application/demo_manifest.json）
   └─ 推理：    HandleBuilder 按 manifest 的 type 构造真实 SDK 模型（ORT CPU）
```

流程：**列出目录 → 选中/加载（懒加载，loading→ready）→ 上传图片推理 → 卸载/切换**。
单槽实例化：加载新模型会先把当前活跃模型卸载（状态回 `unloaded`）；infer 遇到非 `ready`
模型会同步触发一次加载，仍不就绪则返回 503 错误（不自动重试）。

## 构建

根 `CMakeLists.txt` 的 `BUILD_WEB_DEMO`（默认 `OFF`，需显式 `=ON`）门控 demo_server 与
web 资产。CPU 构建示例：

```bash
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON \
      -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF \
      -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_WEB_DEMO=ON
cmake --build build --target demo_server --parallel
```

> MSVC 请用 x64 Native Tools 命令提示符（或先 `call vcvars64.bat`）。
> 资产会自动 `POST_BUILD` 拷到 `build/bin/web_demo/`（含 `index.html`、`samples/`、
> `demo_manifest.json`、`demo_labels/`）。

## 运行

**必须从仓库根目录运行**（资源根由 manifest 的 `base` 字段决定，为仓库根相对路径）：

```bash
build/bin/demo_server --repo application/demo_manifest.json --web build/bin/web_demo
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | `application/demo_manifest.json` | 手写 manifest 文件路径（目录清单） |
| `--web` | `web_demo` | 静态页面目录 |
| `--port` | `8000` | 监听端口（`0` 为随机空闲端口） |

> **注意**：
> - 端口以 `--port` 为准（`0` 为随机）；启动日志打印
>   **`Demo serving on http://127.0.0.1:<端口>/`**，用该端口在浏览器打开演示页。
> - **`--base` 已不存在**。资源根只看 manifest 顶层 `base` 字段（本仓库为
>   `test_data/test_models/onnx`），OCR 字典来自已提交的
>   `application/demo_labels/ppocr_dict.txt`（仓库根相对，构建时随 web 资产拷贝）。

## REST 接口

### GET /v1/models

返回全部模型目录，每项含 `id/display/version/type/labels/input_size/status/ready/error`。
`status` 语义：

| 状态 | 含义 |
|------|------|
| `unloaded` | 仅注册目录，未实例化（启动初始状态；单槽切换也会卸回此态） |
| `loading` | 后台正在构建（`POST /load` 触发的异步线程） |
| `ready` | 构造成功，可推理 |
| `failed` | 构造失败（缺权重 / 设备不支持等），`error` 给出原因 |

另有 `GET /health`、`GET /readyz`、`GET /metrics`。`readyz` 在"恰有一个活跃 Ready"时返回
`ready`，构建中返回 `loading`（503）。

### POST /v1/models/{id}/load

异步触发加载。立即返回当前状态（通常 `loading`）；随后轮询 `GET /v1/models/{id}` 至
`ready`/`failed`。单槽：加载新 id 会先把旧活跃模型卸载。返回 `{id, status}`。

### POST /v1/models/{id}/unload

卸载模型回 `unloaded`。返回 `{id, status}`。非激活模型卸载为 no-op。

### POST /v1/models/{id}/infer

请求体为 JSON 对象，`image` 为图片的 **base64** 字符串（也接受 `image_path` 路径）：

```json
{ "image": "<base64>" }
```

成功返回 `{ results, duration_ms, model }`；`results` 为逐族结果 JSON，框族（det/seg/pose/
obb/face/lpr）的 `box` 坐标为**绝对像素**（`x/y/width/height`，对应输入图的像素坐标系，供前端
直接叠加渲染）。失败返回 `{ error: { code, message } }`，常见状态码：

| 状态码 | code | 说明 |
|--------|------|------|
| 404 | `MODEL_NOT_FOUND` | id 不存在 |
| 503 | `MODEL_NOT_READY` | 模型不在 `ready`（懒加载后仍未就绪 / 加载中），不自动重试 |
| 400 | `BAD_REQUEST` | JSON 非法、解码失败、或推理返回 false |
| 504 | `TIMEOUT` | 超过请求超时（默认 60s）仍未完成 |
| 429 | `RATE_LIMITED` | 超过限流（默认不限） |

## 前端（application/web_demo/index.html）

单文件、无 CDN、无构建。界面为「观测台 / instrument」风格：左栏按族列出模型（状态徽标），中间为画布，
右栏为结果检视器，底部为**测量读数条**（实时显示指针图像坐标与悬停目标的 `x/y/w/h/score`）。

- **两种视图**：`叠加`（默认，前端用返回 JSON 在原图上自绘框/关键点/掩码/车牌）与 `服务端渲染`
  （显示 SDK `vis_*` 渲染图 `image_b64`）。整图族（sem/depth）自动使用服务端渲染。
- **客户端叠加**：det/face/lpr 矩形；pose 骨架 + 关键点；obb 旋转框；seg 半透明实例掩码 + 框；
  ocr 四边形文本框 + 文字。
- **参数**：工具栏「参数」面板配置每请求结果参数——置信度阈值、Top-K（分类）、最大目标数；
  改后点「推理」生效（服务端对结果 JSON 过滤，前端叠加/列表随之更新）。
- **交互**：画布十字准星跟随指针；悬停命中目标高亮 + 浮层（标签/分数/文字）；点击选中，
  与右侧对象列表双向联动；图层开关（框/关键点/掩码/标签）。
- **图片**：上传本地图片（长边降采样 ≤1600px）或选内置样例（scene / gradient / plate）；可导出
  当前叠加结果为 PNG。
- 选中模型触发懒加载（loading 遮罩轮询至 ready/failed）；推理请求携带 `visualize:true`，
  故服务端渲染图始终可用。

## 模型目录（application/demo_manifest.json）

手写 manifest：顶层 `base`（资源根，仓库根相对的目录）与 `models[]` 数组。每条目字段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识（API 路径用） |
| `display` | 前端显示名 |
| `type` | 族名：`det/cls/seg/pose/obb/sem/depth/ocr/face/lpr` |
| `desc` | 一句话说明 |
| `files.model` | 主 ONNX（相对 `base`；绝对路径亦可用） |
| `files.rec` / `files.cls` | OCR/LPR 的识别 / 方向分类模型（可选） |
| `files.dict` | OCR 字典（优先仓库根相对文件，否则相对 `base`） |
| `labels` | 类别表（数组或指向 txt 的路径；缺省留空） |
| `input_size` | 元数据 `[w, h]`，缺省 `[640, 640]` |

**如何新增一个模型**：复制一条现有条目，改 `id/display/desc`，把 `files.model`（及多文件族的
`rec`/`dict`）指到 `base` 目录下已有的 ONNX 即可；启动后在 `/v1/models` 确认该 id `ready=true`
即接入成功。若权重路径有误 / 类型构造失败，该 id 会显示 `failed` 并给出 `error`，不影响其它族。

## 每族端到端验证

在真实权重下逐族：启动 → 确认该族 `ready` → 选一张有代表性内容的图推理，核对可视化/JSON 非空
且结果合理（框贴合、关键点到位、OCR/车牌文字正确）。

本仓库首轮全族端到端矩阵（CPU · ORT · 各族一张测试图）：

| 族 (id) | 加载 | 推理 | 结论 |
|---------|------|------|------|
| det `yolo11n-det` | ready | 200 | 3 个检测框 |
| cls `yolo26n-cls` | ready | 200 | top-1 标签 722 / 0.50 |
| seg `yolo11n-seg` | ready | 200 | 实例掩码 |
| pose `yolo26n-pose` | ready | 200 | 17 关键点骨架 |
| obb `yolo26n-obb` | ready | 200 | 旋转框非空（1024 输入） |
| sem `yolo26n-sem` | ready | 200 | 逐像素标签 |
| depth `yolo26n-depth` | ready | 200 | 深度图 |
| ocr `ppocr5-mobile` | ready | 200 | 识别出文本（小图 ~4s） |
| face `scrfd-2.5g` | ready | 200 | 1 张人脸 + 5 关键点 |
| lpr `lpr-yolov5` | ready | 200 | 2 个车牌号 + 颜色 |

## 已知限制

- **中文路径**：MSVC/OpenCV 在 Windows 上无法打开非 ASCII 路径的图片/模型，模型与上传图片路径
  请使用纯 ASCII。
- **固定阈值**：各族的置信度/可视化阈值在 SDK 侧固定（demo 不开放请求级参数）。
- **请求级参数**：`/v1/models/{id}/infer` 仅接受 `image`/`image_path` 与透传 `params`，无
  每请求阈值/尺寸控制。
- **后端固定**：demo 统一使用 ORT（`use_ort_backend()`），演示多后端请走 SDK 其它示例。
- **输入尺寸由 manifest 驱动**：demofor 支持 `set_size` 的家族（det/cls/seg/pose/obb/sem/depth/face）
  按 manifest 条目 `input_size` 显式设置预处理器目标尺寸，故非 640 输入模型（如
  `yolo26n-obb` 的 1024×1024）也能正确推理；ocr/lpr 为管线模型，不受此控制。
- **OCR 大图 CPU 超时**：PP-OCRv5 Mobile 在大分辨率图（如 1996×1108）CPU 推理可能超过默认
  60s 请求超时 → 返回 504；后台线程仍会跑完（fire-and-forget），但客户端已先返回。换较小图
  （长边数百像素）可在数秒内完成。
- **懒加载在途**：infer 命中非 `ready` 模型时同步触发一次 load，仍不就绪返回 503，不排队。

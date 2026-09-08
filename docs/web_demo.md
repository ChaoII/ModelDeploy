# Web 模型演示（demo_server）

面向浏览器的模型族演示：一个可执行 `demo_server` 同时提供 `ServingServer` REST 推理与静态 Web 页面，网页里就能上传图片试跑各类模型（检测 / 分类 / 实例分割 / 姿态 / 旋转框 / 语义分割 / 深度 / OCR / 人脸 / 车牌），并把推理结果可视化出来。

## 用途

- **快速体验**：不用写客户端代码，浏览器即可验证各模型族的输入输出与可视化效果。
- **模型验收**：在 GPU + 真实权重下，按下方"每族验证步骤"逐族确认端到端推理正确。
- **教学 / 演示**：一张图展示 SDK 的模型覆盖与 ServingServer 能力。

## 架构

```
浏览器 (web_demo/index.html)
   │  POST /v1/models/{name}/infer   (base64 image)
   │  GET  /v1/models                 (元数据 type/labels/input_size)
   ▼
ServingServer (csrc/serving/server.cpp)
   ├─ 静态托管：GET /  → application/web_demo/ （构建时拷到 build/bin/web_demo）
   ├─ 模型仓库：ModelRepo 扫描 repo/{name}/{ver}/，标签 latest
   └─ 推理：   AsyncModel → 真实 SDK 模型（det/cls/seg/pose/obb/sem/depth/ocr/face/lpr）
```

模型句柄由 `application/demo_server.cpp` 的 `build_demo_handle` 按目录名（族名）选择真实模型类，
每个族都包在 `try/catch` 里：**缺权重 / 无设备 / 初始化失败 → 注册为 `ready=false`（空 infer）占位**，
网页仍列出该族，只是试跑返回"模型未就绪"，绝不因单模型问题拖垮进程。

## 构建

在根 `CMakeLists.txt` 中 `BUILD_WEB_DEMO`（默认 `ON`）门控 demo_server 与 web 资产。CPU 构建示例：

```bash
cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON \
      -DBUILD_CAPI=OFF -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF \
      -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_WEB_DEMO=ON
cmake --build build --target demo_server --parallel
```

> MSVC 请用 x64 Native Tools 命令提示符（或先 `call vcvars64.bat`）。
> 资产会自动 `POST_BUILD` 拷到 `build/bin/web_demo/`。

## 运行

```bash
build/bin/demo_server --web build/bin/web_demo --repo <模型仓库目录> --port 8000
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--web` | `web_demo` | 静态页面目录 |
| `--repo` | `demo_repo` | 模型仓库根目录 `repo/{name}/{ver}/` |
| `--port` | `8000` | 期望绑定端口 |

> **注意**：`ServingServer` 实际绑定的是测试后打印的端口。若指定端口被占用，会退避到临时空闲端口，
> 请以启动日志 `Demo serving on http://127.0.0.1:<实际端口>/` 中**打印的端口**为准在浏览器访问。

## 模型仓库布局（每族）

模型目录 `repo/{name}/{ver}/` 至少要有一个 `model.onnx`（`ModelRepo` 以此判定该版本合法），
`name` 即族名（决定走哪个分支与前端渲染器）。各族的额外文件：

| 目录名 | type | 所需文件（`{ver}/` 下） | 前端渲染器 |
|--------|------|------------------------|-----------|
| `det` | det | `model.onnx` | 检测框 |
| `cls` | cls | `model.onnx` | 分类 Top-K |
| `seg` | seg | `model.onnx` | 实例掩码 |
| `pose` | pose | `model.onnx` | 骨架关键点 |
| `obb` | obb | `model.onnx` | 旋转框 |
| `sem` | sem | `model.onnx` | 语义分割色块 |
| `depth` | depth | `model.onnx` | 深度图 |
| `face` | face | `model.onnx`（SCRFD） | 人脸框 + 关键点 |
| `ocr` | ocr | `model.onnx`(det) + `rec.onnx` + `dict.txt`，可选 `cls.onnx` | 文本框 + 文字 |
| `lpr` | lpr | `model.onnx`(det) + `rec.onnx` | 车牌框 + 车牌号 |

- **多文件族**（ocr/lpr）：`model.onnx` 一律作为 det/主模型，识别模型放 `rec.onnx`；
  OCR 的方向分类 `cls.onnx` 缺失时自动禁用，字典 `dict.txt` 必填。
- **`labels.txt`（可选）**：每族模型目录下放一行一个类别名，会覆盖左侧边栏的类别显示；
  未提供时用内置默认表（det/cls 有默认，其余为空）。

## 前端使用

1. 打开打印的实际端口 `/`。
2. 左侧边栏列出仓库中所有模型（名称 + type + 输入尺寸 + 就绪状态）。
3. 选择一族 → 上传本地图片 → 点"推理"，页面显示原图与叠加的渲染结果。

## 每族 GPU 端到端验证

在 GPU + 真实权重下逐族验证（把 demo_server 里 `opt.set_device(Device::CPU, 0)` 改为
`Device::GPU`，或用 `WITH_GPU=ON` 构建并按需改 `Device`）：

1. 放入对应族的 ONNX 权重（建议先从 ultralytics 导出 `yolo11n/…` 与 `ppocr` 系列）。
2. 启动后确认 `/v1/models` 中该族 `ready=true`。
3. 上传一张有代表性内容的图片，核对：det/seg/pose/obb/sem/depth/face 看叠加框/掩码/关键点是否贴合；
   ocr 看文字识别是否准确；lpr 看车牌号 `car_plate_str` 是否正确。
4. 再放一张无目标的图，确认空结果也能正常返回（不报错）。

## 已知限制

- **中文路径**：MSVC/OpenCV 在 Windows 上无法打开非 ASCII 路径的图片/模型，模型仓库与上传图片路径请使用纯 ASCII（如 `C:\models` 与 `model.onnx`），否则 imread/加载会失败并注册为 not-ready。
- **固定阈值**：各族的置信度/可视化阈值在 SDK 侧固定（demo 不开放请求级参数），不支持运行时调参。
- **请求级参数**：`/v1/models/{name}/infer` 仅接受 `image` / `image_path` 与透传 `params`，无每请求阈值/尺寸等控制。
- **后端固定**：demo 统一使用 ORT（可用 `use_ort_backend()`），演示多后端请走 SDK 其它示例。
- **就绪即真实推理**：缺权重的族显示未就绪，网页不可试跑（按设计）。

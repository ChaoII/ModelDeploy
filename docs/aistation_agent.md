# AIStation 布控 Agent（application/aistation_agent）

本文档描述 ModelDeploy 仓库中新增的 **AIStation 布控 Agent**（导出为 `aistation_agent` 可执行文件）：
它面向「云边协同视频分析」场景，在边缘设备或服务器上按云端下发的配置执行
**RTSP/本地视频解码 → 推理 → （可选）编码/快照 → 检测事件上报 + 能力心跳**。

> 阅读前提：了解 SDK 视频模块（[视频编解码总览](./video/README.md)、[接口参考](./video/api.md)）、
> `ImageData` 零拷贝模型（[预处理详解](./preprocess.md)）、后端选择（[后端详解](./backends.md)）
> 以及现有安防应用架构（[AI 智能安防监控平台](./surveillance.md)）。

---

## 1. 定位：为什么不是直接改 `surveillance`

`surveillance`（`application/`）是通用参考应用，负责「本机多路布控 + Web UI」；
AIStation 需要的是「云端下发配置 / 边缘执行 / 事件回传」的**专用边缘节点**。
因此本仓库新增 `application/aistation_agent/`，并**复用** `surveillance` 的管线组件：

- 复用：`TaskConfig`/`ModelConfig` 等配置、`PipelineManager`、`Pipeline`（解码→推理→绘制→编码）、
  `InferGroup`/`InferenceEngine`、`runtime_factory`、SDK `csrc/video` 与 `csrc/vision`。
- 新增：控制面 REST、AIStation 配置适配、检测事件钩子、事件发布（HTTP/MQTT）+ 边缘缓存、
  能力心跳、模型 URL 拉取。
- 原则：**不改 `surveillance` 的源文件行为与测试**；对共享源只做纯增量（默认为空的可选回调/字段）。
  共享组件抽为 CMake **OBJECT 库 `app_common`**，`surveillance` 与 `aistation_agent` 共同链接。

---

## 2. 架构与数据流

```
application/aistation_agent/
  main.cpp                CLI 解析 + 启动 AgentRuntime
  agent_runtime.{h,cpp}   装配：PipelineManager + AgentServer + EventBus + 发布链 + Heartbeat
  agent_server.{h,cpp}    控制面 REST（httplib）
  config_adapter.{h,cpp}  AIStation TaskConfig(JSON) → SDK TaskConfig / ModelConfig / 事件元数据
  model_fetcher.{h,cpp}   模型 URL 拉取（local / http(s) / s3 path-style）到本地缓存
  event_bus.{h,cpp}       检测回调注册/分发、label 维度节流、事件装配（uuid/ts/归一化 bbox）
  event_publisher.{h,cpp} EventPublisher 接口 + HttpPublisher + MqttPublisher + DurableQueue
  capability.{h,cpp}      能力探测（后端/模型族/路数/编解码/硬件）
  heartbeat.{h,cpp}       周期心跳上报（默认 30s，失败指数退避）
```

单路任务的数据流与 `surveillance` 一致（每路一条 `Pipeline`），额外增加事件通道：

```
AIStation ──HTTP 控制面──▶ AgentServer ──ConfigAdapter──▶ PipelineManager ──▶ Pipeline
                                                                               │ 解码→推理→绘制→编码
                                                                               ▼
                            EventBus ◀── Pipeline 检测回调（默认为空）
                               │ 节流 + 装配 DetectionEvent
                               ▼
                        DurableQueue（每任务，落盘） ──worker──▶ MQTT / HTTP ──▶ 云端
Capability ──▶ Heartbeat ──HTTP──▶ {cloud_url}/api/v1/video/edge/heartbeat
```

线程模型：

| 组件 | 线程 | 说明 |
|------|------|------|
| 解码 + `detect_loop` | 每路 | 复用 `Pipeline`（SDK 解码异步 + 单检测线程） |
| 编码（预览） | SDK 异步 | `VideoSink::encode_async`，不占应用线程 |
| 控制面 REST | httplib 线程池 | 任务 CRUD/启停/统计/快照 |
| 事件发布 | 每任务 1 个 `DurableQueue` worker | 与检测线程解耦，不阻塞推理 |
| 心跳 | 1 个线程 | 周期上报，不阻塞布控 |

**关键设计**：检测事件通过 `Pipeline` 新增的**默认为空**回调 `DetectionSink` 产出；
`surveillance` 不设置该回调，因此行为与之前**逐位一致**。

---

## 3. 技术栈与协作方式

| 技术 | 在本 Agent 中的用途 | 如何配合 |
|------|--------------------|----------|
| **ModelDeploy SDK**（`csrc/video`、`csrc/vision`） | 解码/编码、检测推理（`UltralyticsDet`）、绘制、`ImageData` 零拷贝 | `Pipeline`/`VideoSource`/`VideoSink` 薄封装；后端经 `runtime_factory` 由 `backend`/`device` 选择 |
| **OnnxRuntime**（`.onnx`） | 一期默认推理后端（CPU/GPU） | `ModelConfig.backend="ort"`；一期末验证就以 ORT 一条线跑通 |
| **TensorRT / MNN / ncnn / Sophgo** | 按设备能力切换后端 | `ModelConfig.backend`/`device`；由 `capability` 上报的 `backends` 决定云端可下发哪些 |
| **FFmpeg / GStreamer**（经 SDK 视频模块） | RTSP 解码、FLV/RTMP/RTSP 编码推流 | `DecoderConfig`/`EncoderConfig` 透传；`preview.enabled` 控制是否推流 |
| **cpp-httplib**（header-only，`application/third_party`） | ① 控制面 REST 服务 ② `HttpPublisher` 事件上报 ③ `Heartbeat` 上报 ④ `ModelFetcher` HTTP 下载 | 统一使用仓库自带的 header，单进程单实现，无额外系统依赖 |
| **paho.mqtt.c**（`ENABLE_MQTT=ON`，`cmake/paho_mqtt.cmake` 用 `FetchContent` 拉取） | `MqttPublisher`：QoS1 事件上报、稳定 `client_id`、keepalive、断线重连 | 纯 C + CMake，Windows/Linux/Jetson 交叉编译友好；`ENABLE_MQTT=OFF` 时为空实现 |
| **nlohmann-json** | 配置解析、事件/心跳载荷序列化 | 全 Agent 统一 JSON 编解码 |
| **文件缓存队列 `DurableQueue`** | 边缘断网时的本地事件缓冲与恢复补发 | 每任务一个目录；非阻塞入队 + 独立 worker；云端按 `event_id` 去重 |
| **Catch2** | 单元/集成测试 | `application/aistation_agent/tests/`，`aistation_agent_test` 目标 |

### 事件从检测到上云的完整链路

1. `Pipeline::detect_loop()` 推理后，若检测到目标且已注册回调，则把
   `DetectionResult`（像素坐标）转成 `DetectionBox` 并回调。
2. `EventBus::on_detections()` 按 **label 维度**执行 `alarm_interval_sec` 节流，
   生成 `DetectionEvent`（`event_id`=UUID v4、`ts`=UTC ISO8601 毫秒、bbox 归一化、`schema_version=1`）。
3. `AgentRuntime` 把事件按 `task_id` 路由到该任务的 `DurableQueue::enqueue()`（写盘 + 入队，快速返回）。
4. `DurableQueue` 的 worker 按序调用传输层：`HttpPublisher` 或 `MqttPublisher`；
   成功则删除队列文件，失败则保持队首退避重试。
5. 进程重启后 `DurableQueue` 扫描目录继续补发，保证「断网不丢」；重复由云端按 `event_id` 幂等去重。

---

## 4. 快速开始

### 4.1 构建

```bash
# CPU 构建（开启 Agent + 测试 + MQTT）
cmake -S . -B build -G Ninja \
      -DBUILD_AISTATION_AGENT=ON -DBUILD_AISTATION_AGENT_TESTS=ON -DENABLE_MQTT=ON \
      -DBUILD_VISION=ON -DBUILD_VIDEO=ON -DENABLE_ORT=ON -DWITH_GPU=OFF
cmake --build build --target aistation_agent --parallel
```

- `BUILD_AISTATION_AGENT=ON` 会强制 `BUILD_VIDEO`/`BUILD_VISION=ON`，未选视频后端时默认 `ENABLE_FFMPEG=ON`。
- `ENABLE_MQTT=ON` 时经 FetchContent 拉取并构建 paho.mqtt.c 静态库；不需要 MQTT 时可关闭。

### 4.2 运行

```bash
./build/bin/aistation_agent \
  --host 0.0.0.0 --port 19090 \
  --api-key <控制面密钥> \
  --cloud-url http://<cloud> --edge-code edge-01 --secret <心跳 token> \
  --model-cache-dir data/model_cache \
  --max-channels 8
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` / `--port` | `0.0.0.0` / `19090` | 控制面监听地址 |
| `--api-key` | 空 | 非空启用 `Authorization: Bearer <api-key>` 保护 `/api/v1/*` |
| `--cloud-url` | 空 | 心跳基址；空则不发送心跳 |
| `--edge-code` | `edge-01` | 边缘设备标识（进事件与主题） |
| `--secret` | 空 | 心跳 `token`（空时回退 `--api-key`） |
| `--model-cache-dir` | `data/model_cache` | 远端模型下载缓存目录 |
| `--s3-endpoint` | 空 | `s3://` 模型的兼容端点（MinIO/RustFS 等，path-style） |
| `--max-channels` | 8 | 上报能力中的最大并发路数 |
| `--heartbeat-interval` | 30 | 心跳周期（秒） |
| `--data-dir` | 空 | 预留（当前未消费） |

### 4.3 下发一个任务（示例）

```bash
curl -s -X POST http://127.0.0.1:19090/api/v1/tasks \
  -H "Authorization: Bearer <控制面密钥>" -H "Content-Type: application/json" \
  -d '{
    "task_id": 123,
    "camera": {"id": 7, "name": "北门", "url": "rtsp://cam/1", "transport": "tcp"},
    "models": [{
      "name": "aistation-det", "type": "det", "backend": "ort", "device": "cpu",
      "url": "/abs/path/yolo11n_nms.onnx",
      "labels": ["person", "car"], "input_size": [640, 640],
      "confidence_threshold": 0.45
    }],
    "roi": [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]],
    "alarm_interval_sec": 30,
    "algorithm_type": "INTRUSION",
    "preview": {"enabled": false},
    "events": {
      "transport": "mqtt",
      "mqtt": {"broker": "tcp://edge-broker:1883",
               "topic": "aistation/default/edge/edge-01/camera/7/detect", "qos": 1},
      "buffer": {"dir": "./events_buffer", "max_mb": 512}
    }
  }'
curl -s -X POST http://127.0.0.1:19090/api/v1/tasks/123/start -H "Authorization: Bearer <控制面密钥>"
curl -s http://127.0.0.1:19090/api/v1/tasks/123/snapshot.jpg -o snap.jpg -H "Authorization: Bearer <控制面密钥>"
```

---

## 5. 控制面 REST 接口

`AgentServer`（默认端口 `19090`）；统一错误体 `{ "error": { "code": "...", "message": "..." } }`，
状态码 `200/400/401/404/503`。`Authorization: Bearer <api-key>` 非空时保护 `/api/v1/*`；
`/health`、`/readyz` 放行。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | `{ok:true, status:"ok"}`（进程在线） |
| GET | `/readyz` | 200 / 503（有任务卡在 Loading/Failed 时 503 `MODEL_NOT_READY`） |
| GET | `/api/v1/metrics` | 在跑路数 `running_channels`、事件队列 `event_queue_len`、丢弃计数 `event_dropped_total` |
| POST | `/api/v1/tasks` | 下发 AIStation TaskConfig；返回 `{ok:true, task_id}`；配置/模型拉取失败→400 |
| GET | `/api/v1/tasks` | 列出任务状态 |
| GET | `/api/v1/tasks/:id` | 单任务状态 |
| POST | `/api/v1/tasks/:id/start` / `stop` | 启停；不存在→404，启动失败→503 |
| PUT | `/api/v1/tasks/:id` | 先停后改配置 |
| DELETE | `/api/v1/tasks/:id` | 删除任务（并停止其发布链） |
| GET | `/api/v1/tasks/:id/stats` | 透传 `PerfStats`（FPS/丢弃/各阶段延迟） |
| GET | `/api/v1/tasks/:id/snapshot.jpg` | 最新帧 JPEG（复用 `PipelineManager::get_task_jpeg`） |

---

## 6. AIStation TaskConfig → SDK 映射

| AIStation 字段 | 映射 |
|---|---|
| `task_id`(number) | `TaskConfig.id = to_string`；事件 `task_id` |
| `camera.url` / `camera.transport` | `input_url` / `decoder.rtsp_transport` |
| `camera.id` | 事件 `camera_id` |
| `models[].name/type/backend/device` | `ModelConfig`（`type` 支持 `det`/`cls`/`face` 归一化） |
| `models[].url` | 经 `ModelFetcher` 拉到本地 → `ModelConfig.path` |
| `models[].labels` / `input_size` / `confidence_threshold` | `ModelConfig` 对应字段（**以 `confidence_threshold` 为准**） |
| `roi`（归一化多边形） | 取外接矩形 → `ModelConfig.roi_norm`；`InferGroup` 运行时按帧宽高换算像素并 `crop` |
| `alarm_interval_sec` | 事件按 label 节流间隔 |
| `preview.enabled/format` | `TaskConfig.enable_preview` + `output_url`/`preview_url` |
| `decoder.*` / `encoder.*` | `DecoderConfig` / `EncoderConfig` 透传 |
| `events.*` | 传输选择（mqtt/http）、broker/topic/qos/client_id、http url/token、buffer dir/max_mb |
| `algorithm_type` | 原样写入事件 |

---

## 7. 事件契约（Agent → 云端）

**MQTT 主题**：`aistation/{tenant}/edge/{edge_code}/camera/{camera_id}/detect`（`tenant` 默认 `default`）；
**HTTP**：同一 JSON POST 到 `events.http.url`，头 `Authorization: Bearer {token}`。

```jsonc
{
  "event_id": "uuid",                 // 云端据此幂等去重
  "edge_code": "edge-01",
  "camera_id": 7, "task_id": 123,
  "algorithm_type": "INTRUSION",
  "ts": "2026-09-12T08:00:00.123Z",   // UTC ISO8601 毫秒
  "detections": [
    { "label": "person", "label_id": 0, "confidence": 0.91,
      "bbox": { "x": 0.1, "y": 0.2, "width": 0.15, "height": 0.3 } }  // 归一化
  ],
  "latency_ms": 12.3,
  "snapshot": { "ref": "edge-01/cam7/2026-09-12/xxxx.jpg" },          // 可省略
  "schema_version": 1
}
```

---

## 8. 能力探测与心跳

`capability.detect()` 依据编译宏（`ENABLE_ORT/MNN/TRT/NCNN/SOPHGO`、`WITH_GPU`、`ENABLE_VAAPI`）
与运行环境（Jetson `/etc/nv_tegra_release`）产出能力清单；Agent 启动后（默认）每 30s：

`POST {cloud_url}/api/v1/video/edge/heartbeat`

```jsonc
{ "edge_code":"edge-01", "token":"...",
  "capabilities": {
    "hardware": { "platform":"nvidia|cpu|sophgo|jetson", "gpu_model":"...", "vram_mb":0 },
    "backends": ["ort","mnn","trt","ncnn","sophgo"],
    "model_families": ["det","cls","face"],
    "max_channels": 8,
    "codecs": { "decode":["h264","h265"], "encode":["h264"], "hw":["nvenc"] }
  },
  "metrics": { "running_channels": 2, "event_queue_len": 0, "event_dropped_total": 0 },
  "version": "x.y.z" }
```

失败指数退避重试，不阻塞布控。

---

## 9. 边缘离线缓存与去重

- 每任务一个 `DurableQueue`：`<buffer.dir>/<task_id>/`，文件名为 `<seq>-<event_id>.json`。
- `enqueue` 序列化+原子写盘后立即返回（检测线程不阻塞）。
- worker 按序发送；成功删文件，失败保持队首退避重试（保证顺序）。
- 目录超过 `buffer.max_mb` 时删除最旧文件并计入 `event_dropped_total`。
- 进程重启扫描目录从最小序号继续补发；同 `event_id` 不重复入队；云端按 `event_id` 幂等去重。

---

## 10. 模型拉取（ModelFetcher）

| URL 形式 | 行为 |
|----------|------|
| 本地路径 / `file://` | 校验存在后原样使用 |
| `http(s)://` | 下载到 `--model-cache-dir`（命中缓存复用；写临时文件后原子替换） |
| `s3://bucket/key` | 需 `--s3-endpoint`，path-style GET；未配置则明确报错（不引 AWS SDK） |

---

## 11. 预览与快照

- **快照**：`GET /api/v1/tasks/:id/snapshot.jpg` 复用 `Pipeline::latest_jpeg`，任何任务可取最新帧。
- **预览**：`preview.enabled=true` 时 `Pipeline` 通过 SDK 编码路径推 `output_url`（如 FLV）；
  一期 `output_url` 为占位相对路径，实际对外播放需与 AIStation/媒体服务联调（见 §13）。

---

## 12. 构建选项与依赖

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_AISTATION_AGENT` | OFF | 构建 Agent（强制 `BUILD_VIDEO`/`BUILD_VISION=ON`） |
| `BUILD_AISTATION_AGENT_TESTS` | OFF | 构建 `aistation_agent_test` |
| `ENABLE_MQTT` | OFF | 启用 paho.mqtt.c；OFF 时 `MqttPublisher` 为空实现 |

依赖：`app_common`（`surveillance` 与 Agent 共用的 OBJECT 库）、cpp-httplib、nlohmann-json、
OpenCV/FFmpeg（经 SDK 解析）、paho.mqtt.c（`ENABLE_MQTT=ON` 时 FetchContent）、Catch2（测试）。

---

## 13. 部署模式与真实验收

- **云边协同（边缘）**：Agent 跑在边缘设备（Jetson/工控机/GPU 盒子），事件走 MQTT（断网本地缓存补发），
  云端下发配置并订阅事件。
- **纯云端**：Agent 与 AIStation 同网，事件可用 HTTP 或 MQTT。

真实验收可用 Docker 起一个 mosquitto：

```bash
docker run -d --name md-mqtt -p 1883:1883 eclipse-mosquitto:2 \
  sh -c "printf 'listener 1883\nallow_anonymous true\n' > /mosquitto/config/mosquitto.conf && mosquitto -c /mosquitto/config/mosquitto.conf"
# 订阅端（Python 示例）
python -c "import paho.mqtt.client as m; c=m.Client(); c.connect('127.0.0.1',1883); c.subscribe('aistation/#'); c.loop_forever()"
```

然后下发一个 `events.transport="mqtt"` 的任务，观察订阅端收到事件。

---

## 14. 测试

```bash
# 全部 Agent 用例 + surveillance 回归（工作目录由 CTest 设为仓库根）
cd build && ctest -C Release -R "aistation_agent_test|surveillance_test" --output-on-failure
# 或直接运行：
./build/bin/aistation_agent_test              # 全部
./build/bin/aistation_agent_test "[agent][e2e]"   # 端到端
```

覆盖：ConfigAdapter 映射、EventBus 节流/装配、DurableQueue 离线/补发/超限/去重、
capability、heartbeat、HTTP/MQTT 发布（进程内 mock broker）、AgentServer 生命周期、
以及「本地视频 + ONNX det → HTTP/MQTT 事件 + 快照」的端到端。

---

## 15. 已知限制与二期

- 一期以 **ONNX(ORT)** 一条线验收；TRT/MNN/ncnn/Sophgo 按 `backends` 选择。
- `s3://` 仅支持 `--s3-endpoint` 兼容端点，无 AWS SDK。
- `roi` 一期取多边形**外接矩形**；精确多边形 mask 列二期。
- 预览 `output_url` 为占位；与 AIStation/媒体服务联调见二期。
- `capability` 的 `model_families`/`codecs` 目前为静态清单，`gpu_model`/`vram_mb` 为占位。
- 二期：控制面 WebSocket/MQTT 双向、模型热更新、按能力自动改派。

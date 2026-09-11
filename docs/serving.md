# ModelDeploy Serving —— 嵌入式 HTTP 推理网关指南

`ServingServer`（`csrc/serving/server.h`）是 **ModelDeploy 内置的嵌入式 HTTP 推理网关**，把本地模型目录暴露为 REST 端点，免去自行实现 HTTP 服务与调度的成本。典型用法：在同一 C++ 进程里加载 SDK，作为推理服务对外提供 HTTP 接口（对标 `llama.cpp` server 的轻量自托管形态）。

> **本期仅 C++**：`ServingServer` 只在 C++ SDK（`csrc/serving/`）中，尚未接入 Python / C-API / C# 绑定。**gRPC 为二期计划**，本期仅提供 REST。

---

## 1. 特性总览

| 能力 | 说明 |
|------|------|
| 模型目录 | 手写 manifest（`base` + `models[]`）；启动只读元数据、**不实例化** |
| 单槽懒加载 | 同一时刻至多一个模型 `Ready`；`/v1/models/:id/load` 或首次 `/infer` 触发实例化 |
| 内置模型构造 | SDK 内置通用 `HandleBuilder`，按 manifest `type` 构造 det/cls/seg/pose/obb/sem/depth/face/ocr/lpr（需 `BUILD_VISION=ON`） |
| REST 端点 | `health / readyz / metrics / v1/models`、`/v1/models/:id`、`/load`、`/unload`、`/infer` |
| 统一错误体 | `{ "error": { "code": ..., "message": ... } }` |
| 鉴权 | Bearer Token（恒定时间比较），`api_keys` 非空启用 |
| 推理超时 | 单次推理超时返回 `504 TIMEOUT`（底层 AsyncModel worker 继续跑完，不阻塞、不泄漏） |
| 限流 | 全局令牌桶，超限返回 `429 RATE_LIMITED`（`rate_limit_qps > 0` 时） |
| CORS | 跨域头 + `OPTIONS` 预检（`enable_cors`，默认开） |
| Metrics | `/metrics` Prometheus 文本：每模型每状态码请求计数、推理耗时（**有界窗口**） |
| 访问日志 | 请求级访问日志（`enable_access_log`，默认开） |
| 优雅停机 | 停监听 → join 工作线程（handler 内同步完成）→ 停 AsyncModel |
| TLS | HTTPS（`BUILD_SERVING_TLS=ON` + OpenSSL，走 httplib `SSLServer`，可选） |

> **不再支持**（旧设计遗留，已移除）：`{name}/{version}` 目录布局、`latest` 解析、周期性热更新。模型目录由 manifest 显式声明。

---

## 2. 构建开关

`ServingServer` 使用 vendored 的 `cpp-httplib`（头文件）与 `nlohmann/json`，随 SDK 编译。

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_SERVING` | **ON** | 是否把 `csrc/serving/**` 编入 SDK；关闭时不编译 serving 源与 `test_serving`、`demo_server` |
| `BUILD_SERVING_TLS` | OFF | 开启 HTTPS，**需要 OpenSSL**；`find_package(OpenSSL)` 仅在该分支执行 |

```bash
# 常规（不含 TLS）
cmake -S . -B build -G Ninja -DBUILD_SERVING=ON ...
cmake --build build --parallel

# 关闭 serving（不编入网关）
cmake -S . -B build -G Ninja -DBUILD_SERVING=OFF ...
```

> Windows 上安装 OpenSSL 后需确保 CMake 能找到它（可设 `OPENSSL_ROOT_DIR`）。`BUILD_SERVING_TLS=ON` 时定义 `MODELDEPLOY_SERVING_TLS` 与 `CPPHTTPLIB_OPENSSL_SUPPORT`，TLS 装配分支（`httplib::SSLServer`）才参与编译。

---

## 3. 模型目录（manifest）

`ServingConfig::model_repo` 指向一个 **manifest JSON**。`base` 为资源根目录，`models[]` 描述每个模型：

```json
{
  "base": "test_data/test_models/onnx",
  "models": [
    {
      "id": "det",
      "display": "目标检测",
      "type": "det",
      "files": { "model": "yolo/det.onnx" },
      "input_size": [640, 640],
      "labels": ["person", "car"]
    },
    {
      "id": "ocr",
      "display": "文字识别",
      "type": "ocr",
      "files": { "model": "ocr/det.onnx", "cls": "ocr/cls.onnx", "rec": "ocr/rec.onnx", "dict": "ocr/dict.txt" }
    }
  ]
}
```

- `type` 决定构造哪一族模型（见上表内置构造）；`files.model / rec / cls / dict` 为相对 `base` 的路径。
- **启动零实例化**：`scan()` 只读元数据，所有模型 `status=Unloaded`。
- **单槽懒加载**：`/load` 或首次 `/infer` 时构造；加载新模型会先卸下旧模型（同一时刻至多一个 `Ready`）。
- **构造失败不崩进程**：失败模型的 `infer` 为空，`status=Failed`，`/infer` 返回 `503 MODEL_NOT_READY` 并附 `error`。
- 可注入自定义 `HandleBuilder` 覆盖内置构造（如换 GPU 设备、特殊前后处理）。

覆盖模型族（内置）：`det / cls / seg / pose / obb / sem / depth / face / ocr / lpr`。

---

## 4. 端点一览

BASE 由绑定地址与端口决定（`host` / `port`）。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活：进程在线且目录已扫描 → `200`（**不要求有模型 Ready**） |
| GET | `/readyz` | 就绪：无模型卡在 `Loading`/`Failed` → `200`，否则 `503` |
| GET | `/metrics` | Prometheus 文本（默认需鉴权，见 §5.1） |
| GET | `/v1/models` | 列出全部模型（id/display/type/labels/input_size/status/ready/error） |
| GET | `/v1/models/:id` | 查询指定模型状态 |
| POST | `/v1/models/:id/load` | 同步加载（构造模型）；返回当前状态 |
| POST | `/v1/models/:id/unload` | 卸载当前活跃模型 |
| POST | `/v1/models/:id/infer` | 推理，请求体见下 |

### 推理请求体

```json
{
  "image": "<base64 编码图像>",
  "visualize": true,
  "params": { "threshold": 0.5 }
}
```

- `image`（base64）或 `image_path`（本地路径）二选一。
- `visualize`（可选，默认 `false`）：为 `true` 时额外返回标注图 `image_b64`（base64 JPEG）与 `image_w/image_h`（需 manifest 模型带可视化，且 `font_path` 非空才能画中文）。
- `params`（可选）：原样回传。

成功响应（示例）：

```json
{ "model": "det", "results": [ ... ], "duration_ms": 12.3 }
```

失败统一为：

```json
{ "error": { "code": "MODEL_NOT_FOUND", "message": "model not found: nope" } }
```

### 状态码

| 状态码 | code | 触发 |
|--------|------|------|
| 200 | — | 推理成功 |
| 400 | `BAD_REQUEST` | JSON 非法 / 非对象 / 推理失败 |
| 401 | `UNAUTHORIZED` | 缺/错 Bearer Token（启用鉴权时，含 `/metrics`） |
| 404 | `MODEL_NOT_FOUND` | 模型不存在 |
| 413 | `PAYLOAD_TOO_LARGE` | 请求体超过 `max_body_bytes` |
| 429 | `RATE_LIMITED` | 超出全局限流（`rate_limit_qps > 0`） |
| 503 | `MODEL_NOT_READY` | 模型未就绪 / 加载失败 |
| 504 | `TIMEOUT` | 单次推理超时（`request_timeout`） |

---

## 5. 配置

通过 `ServingConfig` 装配 `ServingServer`：

```cpp
#include "serving/config.h"
#include "serving/server.h"

modeldeploy::serving::ServingConfig cfg;
cfg.host = "0.0.0.0";               // 绑定地址
cfg.port = 8000;                    // 0=随机端口；>0 绑定指定端口
cfg.model_repo = "application/demo_manifest.json";  // manifest 路径
cfg.web_root = "web_demo";          // 非空时同源托管该静态目录
cfg.font_path = "test_data/msyh.ttc"; // 可视化字体（空=不绘制中文）
cfg.http_threads = 0;               // 0=硬件并发；>0 指定工作线程数
cfg.api_keys = {"sk-xxxx"};         // 非空则启用 Bearer 鉴权
cfg.max_body_bytes = 64 << 20;      // 请求体上限
cfg.request_timeout = std::chrono::milliseconds(60000);  // 推理超时(504)
cfg.rate_limit_qps = 0;             // 0=不限流；>0 启用全局令牌桶(429)
cfg.enable_cors = true;             // CORS 默认开
cfg.metrics_require_auth = true;    // /metrics 是否需鉴权（有 api_keys 时）
cfg.enable_access_log = true;       // 请求级访问日志

// TLS（仅 BUILD_SERVING_TLS=ON 时生效）
cfg.enable_tls = true;
cfg.tls_cert = "server.crt";
cfg.tls_key  = "server.key";

modeldeploy::serving::ServingServer server(cfg);   // nullptr builder → 内置通用构造
std::string err;
if (!server.start(&err)) { /* 处理绑定/TLS 失败 */ }
// ... 运行 ...
server.stop();
```

### 5.1 鉴权

`api_keys` 非空时全局启用（含 `/metrics`，可经 `metrics_require_auth=false` 放开）：

```
Authorization: Bearer <key>
```

校验为恒定时间比较，防时序侧信道。

### 5.2 限流

`rate_limit_qps > 0` 时启用**全局令牌桶**：按恒定速率补充令牌（容量=1，无突发窗口），超限返回 `429 RATE_LIMITED`（统一错误体）。默认 `0` 不限流。

### 5.3 CORS

`enable_cors`（默认 `true`）时，带 `Origin` 的请求自动追加：

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET,POST,OPTIONS
Access-Control-Allow-Headers: Authorization,Content-Type
```

`OPTIONS` 预检直接回 `204`（关闭 CORS 时预检回 `404`）。

### 5.4 TLS

`BUILD_SERVING_TLS=ON` 且 `enable_tls && tls_cert` 非空时，用 `httplib::SSLServer` 提供 HTTPS；否则普通 HTTP。TLS 装配分支在 `#if defined(MODELDEPLOY_SERVING_TLS)` 下编译。

### 5.5 优雅停机

`stop()` 内按顺序：关闭监听 socket → join 工作线程（handler 内的同步推理会在此完成）→ 关闭后由 `repo`/`AsyncModel` 析构收尾在途任务。可重复调用（幂等）；析构时自动 `stop()`。

---

## 6. Metrics（Prometheus）

`GET /metrics` 输出 Prometheus 文本（`text/plain; version=0.0.4`，默认需鉴权），样例：

```
# HELP modeldeploy_serving_requests_total Number of inference requests served per model and status code.
# TYPE modeldeploy_serving_requests_total counter
modeldeploy_serving_requests_total{model="det",code="200"} 5
modeldeploy_serving_requests_total{model="nope",code="404"} 1
# HELP modeldeploy_serving_inference_ms Inference latency in milliseconds.
# TYPE modeldeploy_serving_inference_ms summary
modeldeploy_serving_inference_ms_sum{model="det"} 123.45
modeldeploy_serving_inference_ms_count{model="det"} 5
modeldeploy_serving_inference_ms{model="det",quantile="0.5"} 24.6
modeldeploy_serving_inference_ms{model="det",quantile="0.95"} 31.2
modeldeploy_serving_inference_ms{model="det",quantile="1"} 40.1
```

- `requests_total{model,code}`：每模型每状态码请求计数。
- `inference_ms`：推理耗时聚合（sum/count + 中位/95%/最大分位数）。原始样本按**每模型 1024 的环形窗口**保留（有界），`/metrics` 时聚合输出。
- `modeldeploy_serving_in_flight`（Gauge）：当前在途推理数。
- `modeldeploy_serving_model_load_total{result="ok|fail"}`（Counter）：模型加载尝试次数。
- `modeldeploy_serving_auth_failures_total`（Counter）：鉴权失败次数。
- `modeldeploy_serving_rate_limited_total`（Counter）：限流命中次数。

---

## 7. curl 示例

```bash
HOST=127.0.0.1:8000

curl -s "$HOST/health"
curl -s "$HOST/readyz"
curl -s "$HOST/v1/models"

# 懒加载某模型
curl -s -X POST "$HOST/v1/models/det/load"

# 推理（base64 image body；带可视化）
curl -s -X POST "$HOST/v1/models/det/infer" \
     -H "Content-Type: application/json" \
     -d '{"image":"<base64图像>","visualize":true}'

# 推理（带 Bearer 鉴权）
curl -s -X POST "$HOST/v1/models/det/infer" \
     -H "Authorization: Bearer sk-xxxx" \
     -H "Content-Type: application/json" \
     -d '{"image_path":"/abs/path/to/image.jpg"}'

# 指标（启用鉴权时需带 token）
curl -s "$HOST/metrics" -H "Authorization: Bearer sk-xxxx"
```

---

## 8. 已知约束 / 路线

- **本期仅 C++**：无 Python/C-API/C# 绑定；需跨语言调用请自建薄代理或等 gRPC。
- **gRPC 二期**：当前仅 REST。
- **单槽懒加载**：同一时刻至多一个模型 `Ready`；如需多模型常驻，为未来增强项。
- **TLS 需自备证书**：`tls_cert` / `tls_key` 由外部提供（可用 `openssl req -x509 ...` 自签）。
- 限流为**全局**令牌桶（不按 IP / key 维度）。

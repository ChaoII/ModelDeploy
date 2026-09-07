# ModelDeploy Serving —— 嵌入式 HTTP 推理网关指南

`ServingServer`（`csrc/serving/server.h`）是 **ModelDeploy 内置的嵌入式 HTTP 推理网关**，把本地模型仓库暴露为 REST 端点，免去自行实现 HTTP 服务与调度的成本。典型用法：在同一 C++ 进程里加载 SDK，作为推理服务对外提供 HTTP 接口（对标 `llama.cpp` server / Triton 的轻量自托管形态）。

> **本期仅 C++**：`ServingServer` 只在 C++ SDK（`csrc/serving/`）中，尚未接入 Python / C-API / C# 绑定。**gRPC 接入为二期计划**，本期仅提供 REST。

---

## 1. 特性总览

| 能力 | 说明 |
|------|------|
| 模型仓库 | 目录热扫描：`repo/{model}/{version}/model.onnx`，`latest` 解析，热更新（`hot_reload_interval`） |
| REST 端点 | `health / readyz / metrics / v1/models`、`/v1/models/:name`、`/v1/models/:name/infer` |
| 统一错误体 | `{ "error": { "code": ..., "message": ... } }` |
| 鉴权 | Bearer Token（恒定时间比较），`api_keys` 非空启用 |
| 推理超时 | 单次推理超时返回 `504 TIMEOUT`，后台线程在途跑完不泄漏（fire-and-forget） |
| 限流 | 全局令牌桶，超限返回 `429 RATE_LIMITED`（`rate_limit_qps > 0` 时） |
| CORS | 跨域头 + `OPTIONS` 预检（`enable_cors`，默认开） |
| Metrics | `/metrics` Prometheus 文本：每模型每状态码请求计数、推理耗时 sum/count/分位数 |
| 优雅停机 | 停监听 → 等在途完成（受界）→ 停仓库内 AsyncModel |
| TLS | HTTPS（`BUILD_SERVING_TLS=ON` + OpenSSL，走 httplib `SSLServer`，可选） |

---

## 2. 构建开关

`ServingServer` 使用 vendored 的 `cpp-httplib`（头文件）与 `njson`/`AsyncModel`，默认随 SDK 编译；服务化功能本身不额外引入运行时依赖。

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_SERVING` | OFF（源码经 `GLOB` 恒随 SDK 编入，开关当前不门控编译） | 构建内嵌推理网关（C++） |
| `BUILD_SERVING_TLS` | OFF | 开启 HTTPS，**需要 OpenSSL**；`find_package(OpenSSL)` 仅在该分支执行 |

```bash
# 常规（不含 TLS）
cmake -S . -B build -G Ninja -DBUILD_SERVING=ON ...
cmake --build build --parallel

# 开启 HTTPS（需系统已装 OpenSSL，如 Windows: OpenSSL-Win64）
cmake -S . -B build -G Ninja -DBUILD_SERVING=ON -DBUILD_SERVING_TLS=ON ...
```

> Windows 上安装 OpenSSL 后需确保 CMake 能找到它（可设 `OPENSSL_ROOT_DIR`）。`BUILD_SERVING_TLS=ON` 时定义 `MODELDEPLOY_SERVING_TLS` 与 `CPPHTTPLIB_OPENSSL_SUPPORT`，TLS 装配分支（`httplib::SSLServer`）才参与编译。

---

## 3. 模型仓库布局

默认根目录由 `ServingConfig::model_repo` 指定，按「模型 / 版本」两层组织：

```
repo/
├── det/
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
└── cls/
    └── latest/
        └── model.onnx
```

- `latest`：请求未指定版本时默认解析到最新版本（数字版本按数值比较，`10 > 2`）。
- 热更新：按 `hot_reload_interval` 周期性扫描，新增/变更版本自动纳入（旧持有者句柄仍可完成在途推理）。
- 模型实例由 `HandleBuilder` 创建（默认经 `make_model_handle` 装配进 `AsyncModel` 异步壳，支持攒批与有界背压）；测试可注入 fake 构造器。

---

## 4. 端点一览

BASE 由绑定地址与端口决定（`host` / `port`）。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活检查：任一模型就绪 → `200 {"status":"ok"}`，否则 `503` |
| GET | `/readyz` | 就绪检查：全部模型就绪 → `200`，否则 `503` |
| GET | `/metrics` | Prometheus 文本指标（限流/CORS 同样适用） |
| GET | `/v1/models` | 列出全部模型（name/version/ready） |
| GET | `/v1/models/:name` | 查询指定模型状态 |
| POST | `/v1/models/:name/infer` | 推理，请求体见下 |

### 推理请求体

```json
{
  "image": "<base64 编码图像>"
}
```

或指定本地图像路径：

```json
{ "image_path": "/abs/path/to/image.jpg" }
```

成功响应（示例）：

```json
{ "model": "det", "results": "...", "duration_ms": 12.3 }
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
| 401 | `UNAUTHORIZED` | 缺/错 Bearer Token（启用鉴权时） |
| 404 | `MODEL_NOT_FOUND` | 模型或版本不存在 |
| 429 | `RATE_LIMITED` | 超出全局限流（`rate_limit_qps > 0`） |
| 503 | `MODEL_NOT_READY` | 模型未就绪 |
| 504 | `TIMEOUT` | 单次推理超时（`request_timeout`） |

---

## 5. 配置

通过 `ServingConfig` 装配 `ServingServer`：

```cpp
#include "serving/config.h"
#include "serving/server.h"

modeldeploy::serving::ServingConfig cfg;
cfg.host = "0.0.0.0";               // 绑定地址
cfg.port = 8000;
cfg.model_repo = "repo";            // 模型仓库根
cfg.http_threads = 0;               // 0=硬件并发；>0 指定工作线程数
cfg.api_keys = {"sk-xxxx"};         // 非空则启用 Bearer 鉴权
cfg.max_body_bytes = 64 << 20;      // 请求体上限
cfg.request_timeout = std::chrono::milliseconds(60000);  // 推理超时(504)
cfg.rate_limit_qps = 0;             // 0=不限流；>0 启用全局令牌桶(429)
cfg.enable_cors = true;             // CORS 默认开
cfg.hot_reload_interval = std::chrono::seconds(5);

// TLS（仅 BUILD_SERVING_TLS=ON 时生效）
cfg.enable_tls = true;
cfg.tls_cert = "server.crt";
cfg.tls_key  = "server.key";

modeldeploy::serving::ServingServer server(cfg);
if (!server.start(&err)) { /* 处理绑定/TLS 失败 */ }
// ... 运行 ...
server.stop();
```

### 5.1 鉴权

`api_keys` 非空时全局启用。请求需携带：

```
Authorization: Bearer <key>
```

校验为恒定时间比较，防时序侧信道。

### 5.2 限流

`rate_limit_qps > 0` 时启用**全局令牌桶**：按恒定速率补充令牌（容量=1，无突发窗口），
超限返回 `429 RATE_LIMITED`（统一错误体）。默认 `0` 不限流，不影响既有路径。

### 5.3 CORS

`enable_cors`（默认 `true`）时，带 `Origin` 的请求自动追加：

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET,POST,OPTIONS
Access-Control-Allow-Headers: Authorization,Content-Type
```

`OPTIONS` 预检直接回 `204`（关闭 CORS 时预检回 `404`，不泄露跨域许可）。

### 5.4 TLS

`BUILD_SERVING_TLS=ON` 且 `enable_tls && tls_cert` 非空时，用 `httplib::SSLServer` 提供 HTTPS；
否则普通 HTTP。TLS 装配分支在 `#if defined(MODELDEPLOY_SERVING_TLS)` 下编译，未开 TLS 的构建不含该路径。

### 5.5 优雅停机

`stop()` 内按顺序：关闭监听 socket → join 工作线程（在途同步 handler 完成）→
等等游离 fire-and-forget 推理线程（受 `request_timeout + 1s` 上界）→ 停仓库内 AsyncModel。
可重复调用（幂等）；析构时自动 `stop()`。

---

## 6. Metrics（Prometheus）

`GET /metrics` 输出 Prometheus 文本（`text/plain; version=0.0.4`），样例：

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

- `requests_total{model,code}`：每模型每状态码请求计数（含 `200/404/503/400/429/504`）。
- `inference_ms`：推理耗时聚合（sum/count + 中位/95%/最大分位数），原始样本进程内留存，`/metrics` 时聚合输出。

---

## 7. curl 示例

```bash
HOST=127.0.0.1:8000

# 健康 / 就绪
curl -s "$HOST/health"
curl -s "$HOST/readyz"

# 模型列表
curl -s "$HOST/v1/models"

# 模型状态
curl -s "$HOST/v1/models/det"

# 推理（base64 image body，默认无鉴权）
curl -s -X POST "$HOST/v1/models/det/infer" \
     -H "Content-Type: application/json" \
     -d '{"image":"<base64图像>"}'

# 推理（带 Bearer 鉴权）
curl -s -X POST "$HOST/v1/models/det/infer" \
     -H "Authorization: Bearer sk-xxxx" \
     -H "Content-Type: application/json" \
     -d '{"image":"<base64图像>"}'

# 本地图像路径推理
curl -s -X POST "$HOST/v1/models/det/infer" \
     -H "Content-Type: application/json" \
     -d '{"image_path":"/abs/path/to/image.jpg"}'

# 指标
curl -s "$HOST/metrics"

# 超时示例（假设配了 request_timeout）：会回 504
# 限流示例（rate_limit_qps=1、快速连发）：第二个回 429
```

---

## 8. 已知约束 / 路线

- **本期仅 C++**：无 Python/C-API/C# 绑定；需跨语言调用请自建薄代理或等 gRPC。
- **gRPC 二期**：当前仅 REST；gRPC 网关规划为二期。
- **TLS 需自备证书**：`tls_cert` / `tls_key` 由外部提供（可用 `openssl req -x509 ...` 自签）。
- 限流为**全局**令牌桶（不按 IP / key 维度）；如需每客户端配额可在其上扩展。

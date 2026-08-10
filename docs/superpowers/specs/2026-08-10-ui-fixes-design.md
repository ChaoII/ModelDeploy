# UI 修复优化（子项目 B）— 设计文档

日期：2026-08-10
分支：main

## 背景与动机

NEXUS 多路视频分析应用（`application/`）的 Web 仪表盘（`web_ui.html`，95KB）和 REST 服务（`http_server.cpp`）存在安全问题、功能缺陷、性能问题和缺失功能。基于完整 UI 审查（audit）发现问题清单，本次系统性修复优化。

## 问题清单（来自 UI audit）

### Critical（安全）
1. **存储型 XSS**：`esc()`（web_ui.html:662）不转义单引号 `'`，但内联 onclick 用单引号字符串 → 任务 ID/模型名含 `'` 可注入任意 JS

### Important（功能）
2. **FLV 端口硬编码 8080**（web_ui.html:1435）：media server 端口写死，与 app server 默认 18080 不一致
3. **`chaseLiveLatency` 死代码**（web_ui.html:1178-1191）：`p._player.mediaElement` 永远 undefined
4. **`enable_preview` 被忽略**（web_ui.html:1224）：预览禁用时仍显示 LIVE
5. **任务 ID 未 URL 编码**（web_ui.html:1401/1408/1415）：含特殊字符的 ID 破坏路由
6. **web_ui.html 路径硬编码 + 每请求重读**（http_server.cpp:126, 183-191）
7. **轮询过频**：缩略图 400ms/任务（1270-1293）+ 每 2s 双接口（tasks+metrics）

### Minor（健壮性/体验）
8. 统计重复计数（statStopped 含 error）
9. 空模型列表守卫缺失（updateTask）
10. 编辑弹窗模型预选 selector bug
11. 细节图表不刷新
12. 硬编码默认值（dev 路径、端口）
13. fetch 无错误/加载状态
14. Chart.js/flv.js CDN 依赖（离线不可用）
15. `renderModels` 无 input_size 守卫

### 缺失功能
16. 任务内动态模型管理（PATCH models API 有，UI 没暴露）
17. `avg_total_ms`/`avg_decode_ms` 未显示

## 设计

### 1. XSS 修复（Critical）

**方案**：修复 `esc()` 转义单引号，并加一个专门的 JS 字符串安全函数：
```js
function esc(s){if(s==null)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
// 用于 onclick 内联参数（JS 字符串安全）
function jsStr(s){return String(s==null?'':s).replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/"/g,'\\"')}
```
- `esc()` 用于 HTML 内容转义（补 `'` → `&#39;`）
- `jsStr()` 用于 onclick 内联参数（转义 `\` 和 `'`）
- 所有内联 onclick 的 id/name 参数改用 `jsStr(...)`

### 2. FLV 端口配置（Important）

**方案**：`deriveHttpFlv` 从配置获取媒体服务器端口：
- 服务端注入 `window.MEDIA_SERVER_PORT`（从配置/默认 8080）
- 前端 `deriveHttpFlv` 用该端口而非硬编码
- `http_server` 提供 `/api/v1/config` 返回 media 端口

### 3. 死代码/预览逻辑修复（Important）
- 移除 `chaseLiveLatency` 的 `p._player` 误用，用正确的 `p.mediaElement`
- `liveOn` 加 `&& t.enable_preview` 检查
- 预览禁用时显示"预览已禁用"占位

### 4. URL 编码（Important）
- 所有任务 ID 拼接 URL 用 `encodeURIComponent`
- 内联 onclick 传已编码的 ID

### 5. web_ui.html 路径/缓存（Important）
- `http_server.cpp`：启动时加载一次 web_ui.html 到内存缓存，`load_web_ui()` 返回缓存
- 移除硬编码 `E:\CLionProjects\...` 路径（保留相对路径回退）
- 增加 `--web-ui <path>` 可选参数覆盖

### 6. 轮询优化（Important）
- 缩略图轮询：400ms → **1500ms**，且全局串行（一次一个在途请求）
- `/api/v1/metrics` 与 `/api/v1/tasks` 合并：metrics 从 tasks 响应计算（前端），停止 metrics 轮询
- tasks 轮询 2s → 3s（状态变化不敏感）

### 7. Minor 修复
- `statStopped = total - running - error`（互斥）
- `updateTask` 空模型守卫（与 createTask 一致）
- 编辑弹窗模型预选：用 DOM value 比较而非 CSS selector
- 细节图表每 poll tick 刷新（chart.update）
- 移除硬编码 dev 默认值（input_url/output_url/mf_path）
- fetch 加错误横幅 + loading 状态 + timeout
- Chart.js/flv.js 降级守卫（`typeof Chart==='undefined'` 提示）
- `renderModels` 加 input_size 守卫

### 8. 缺失功能
- **任务详情弹窗**：模型列表加"添加/移除模型"控件（调 PATCH /api/v1/tasks/:id/models）
- **指标显示**：详情网格加 avg_total_ms/avg_decode_ms
- **模型管理**：模型库弹窗加 rec_path/labels 编辑（可选）

## 组件改动清单

| 文件 | 改动 |
|------|------|
| `application/web_ui.html` | 全部前端修复（XSS/FLV/轮询/动态模型/指标） |
| `application/http_server.cpp` | web_ui 缓存、/api/v1/config、动态模型 API 已有（确认暴露） |
| `application/http_server.hpp` | （如需）web_ui 缓存成员 |

## 验证

1. `surveillance_test` 全过（http_server 相关）
2. 启动 app，浏览器验证：
   - 任务创建/编辑/删除无 XSS 注入
   - 预览流正常（FLV 端口配置生效）
   - 20 路任务卡片 FPS 显示正常
   - 任务详情模型管理可用
   - 缩略图刷新不卡顿

## 错误处理

- web_ui.html 加载失败 → 明确报错（已有）
- CDN 不可用 → 降级提示
- fetch 超时 → 错误横幅

## 风险

- web_ui.html 是单文件 95KB 大改动（前端 JS），需仔细回归
- 动态模型管理涉及运行中任务的模型增删（PATCH API 已实现）

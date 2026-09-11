#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace modeldeploy::serving {

struct ServingConfig {
    std::string host = "0.0.0.0";
    int port = 8000;                        // 0=绑定随机端口
    std::string model_repo;                 // manifest 路径
    std::string web_root;                   // 非空时 ServingServer 同源托管该静态目录
    std::string font_path;                  // 可视化字体路径（空=不绘制中文文本）
    size_t http_threads = 0;                // 0=硬件并发
    std::vector<std::string> api_keys;      // 非空启用鉴权
    size_t max_body_bytes = 64 << 20;
    std::chrono::milliseconds request_timeout{60000};
    double rate_limit_qps = 0;              // 0=不限
    bool enable_cors = true;
    bool enable_tls = false;                // 需 BUILD_SERVING_TLS
    std::string tls_cert, tls_key;
    bool metrics_require_auth = true;       // /metrics 是否需鉴权（有 api_keys 时）
    bool enable_access_log = true;          // 请求级访问日志
};

}  // namespace modeldeploy::serving

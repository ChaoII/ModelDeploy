#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace modeldeploy::serving {

struct ServingConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string model_repo;                 // 仓库根
    size_t http_threads = 0;                // 0=硬件并发
    std::vector<std::string> api_keys;      // 非空启用鉴权
    size_t max_body_bytes = 64 << 20;
    std::chrono::milliseconds request_timeout{60000};
    double rate_limit_qps = 0;              // 0=不限
    bool enable_cors = true;
    bool enable_tls = false;                // 需 BUILD_SERVING_TLS
    std::string tls_cert, tls_key;
    std::chrono::seconds hot_reload_interval{5};
};

}  // namespace modeldeploy::serving

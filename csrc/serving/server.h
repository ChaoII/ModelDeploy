//
// ServingServer —— 嵌入式 HTTP 推理网关实现（Task 4 + Task 5 完备化）。
//
// 装配 httplib::Server + ModelRepo，暴露 REST 端点，走统一错误体。能力：
//   路由 / 统一错误体 / Bearer 鉴权（恒定时间） / 推理超时（504）/ 健康检查 / 优雅停机
//   / 令牌桶限流（429）/ CORS / /metrics Prometheus 指标 / TLS（BUILD_SERVING_TLS，#ifdef）。
// 优雅停机：停 listen → join 工作线程（在途同步 handler 完成）→ wait_drained 等游离
// fire-and-forget 推理线程（受 request_timeout + 1s 界）→ 停 repo 内 AsyncModel。
//
#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/md_decl.h"
#include "serving/config.h"
#include "serving/model_repo.h"

namespace httplib {
class Server;
}

namespace modeldeploy::serving {

// 进程内指标（线程安全，mutex 保护）。requests 计各模型各状态码的请求数；
// inference_ms 存原始样本（ms），/metrics 聚合 avg/max/p95。
struct ServingMetrics {
    static constexpr size_t kMaxInferenceSamples = 1024;  // 每模型延迟样本上限（环形窗口）
    mutable std::mutex m;                           // 被 const render() 锁定，须 mutable
    std::map<std::string, std::map<int, uint64_t>> requests;    // model -> (code -> count)
    std::map<std::string, std::vector<double>> inference_ms;    // model -> raw samples (ms, 有界)
    std::atomic<uint64_t> in_flight{0};             // 当前在途推理数（Gauge）
    std::atomic<uint64_t> model_load_ok{0};         // 加载成功次数（Counter）
    std::atomic<uint64_t> model_load_fail{0};       // 加载失败次数（Counter）
    std::atomic<uint64_t> auth_failures{0};         // 鉴权失败次数（Counter）
    std::atomic<uint64_t> rate_limited{0};          // 限流命中次数（Counter）

    void record_request(const std::string& model, int code);
    void record_inference(const std::string& model, double ms);
    std::string render() const;  // Prometheus 文本
};

class TokenBucket;  // 实现见 server.cpp（全局令牌桶）

// HTTP 推理网关。构造可注入 HandleBuilder（测试注入 FakeModel 版句柄构造器）。
class MODELDEPLOY_CXX_EXPORT ServingServer {
public:
    explicit ServingServer(const ServingConfig& cfg, HandleBuilder builder = nullptr,
                           std::string* err = nullptr);
    ~ServingServer();

    ServingServer(const ServingServer&) = delete;
    ServingServer& operator=(const ServingServer&) = delete;

    bool start(std::string* err = nullptr);  // 起 http listen + repo 首次 scan
    void stop();                             // 优雅停机（drain 在途请求）
    bool is_listening() const;
    int port() const;                        // 绑定端口（bind_to_any_port 后有效；测试用）
    ModelRepo* repo();                       // 供测试检查

private:
    void register_routes();
    std::shared_ptr<httplib::Server> make_http_server();

    ServingConfig cfg_;
    std::shared_ptr<ModelRepo> repo_;
    std::shared_ptr<httplib::Server> srv_;
    std::thread listen_thread_;
    int bound_port_ = 0;
    std::atomic<bool> started_{false};
    std::atomic<bool> listening_{false};
    std::shared_ptr<ServingMetrics> metrics_;
    std::unique_ptr<TokenBucket> limiter_;
};

}  // namespace modeldeploy::serving

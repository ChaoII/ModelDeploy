//
// ServingServer —— 嵌入式 HTTP 推理网关骨架（Task 4）。
//
// 装配 httplib::Server + ModelRepo，暴露 REST 端点，走统一错误体。本期实现：
//   路由 / 统一错误体 / Bearer 鉴权（恒定时间） / 推理超时（504）/ 健康检查 / 优雅停机骨架。
// 限流（令牌桶 / CORS / /metrics 完整指标 / TLS）在 Task 5 实现；本任务 /metrics 仅占位。
//
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "core/md_decl.h"
#include "serving/config.h"
#include "serving/model_repo.h"

namespace httplib {
class Server;
}

namespace modeldeploy::serving {

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

    // 在途请求计数：仅统计脱离 httplib 工作线程池的推理 worker（fire-and-forget 超时场景）。
    // httplib 的 ThreadPool::shutdown() 已在 listen 收尾时 join 同步 handler，故此处只需
    // 守候这些游离异步线程，保证优雅停机时不析构仍在用 AsyncModel 的句柄。
    void begin_request();
    void end_request();
    void wait_drained();

    ServingConfig cfg_;
    std::shared_ptr<ModelRepo> repo_;
    std::shared_ptr<httplib::Server> srv_;
    std::thread listen_thread_;
    int bound_port_ = 0;
    std::atomic<bool> started_{false};
    std::atomic<bool> listening_{false};
    std::mutex drain_mtx_;
    std::condition_variable drain_cv_;
    uint64_t in_flight_ = 0;
};

}  // namespace modeldeploy::serving

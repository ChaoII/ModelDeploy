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

// 共享在途计数 + drain 同步状态（UAF 修复核心）。独立堆对象：ServingServer 持一份
// shared_ptr，同时每个 fire-and-forget 推理线程也捕获一份 shared_ptr。这样即便
// ServingServer 已析构（wait_drained 超期返回后 srv_.reset()/析构 drain 原成员），
// 游离线程仍能借 DrainState 安全递减 + notify，不会触及已析构的 this。
struct DrainState {
    std::mutex m;
    std::condition_variable cv;
    size_t in_flight = 0;
};

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
    // 计数/同步写在 drain_（shared_ptr<DrainState>）上，而非本对象成员，以避免
    // 游离线程在 ServingServer 析构后触碰已失效状态（UAF）。
    void begin_request();
    void wait_drained();

    ServingConfig cfg_;
    std::shared_ptr<ModelRepo> repo_;
    std::shared_ptr<httplib::Server> srv_;
    std::thread listen_thread_;
    int bound_port_ = 0;
    std::atomic<bool> started_{false};
    std::atomic<bool> listening_{false};
    std::shared_ptr<DrainState> drain_;
};

}  // namespace modeldeploy::serving

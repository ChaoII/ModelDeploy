//
// ServingServer —— HTTP 推理网关实现（Task 4 骨架）。
//
// 路由/错误体/Bearer 鉴权/超时/健康/优雅停机见 server.h 注释。
// httplib 工作线程池：默认硬件并发（或 cfg_.http_threads）；见 httplib.h。
//
#include "serving/server.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "httplib.h"

namespace modeldeploy::serving {

namespace {

// 恒定时间字符串比较（长度不相等直接失败；恒定只对等长成立，长度泄露通常可接受）。
bool ct_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char diff = 0;
    for (size_t i = 0; i < a.size(); ++i)
        diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    return diff == 0;
}

// 统一错误体：{ "error": { "code": ..., "message": ... } }
void write_error(httplib::Response& res, int status, const std::string& code,
                 const std::string& message) {
    res.status = status;
    res.set_content(nlohmann::json{{"error", {{"code", code}, {"message", message}}}}.dump(),
                    "application/json");
}

// Bearer 鉴权中间件：cfg.api_keys 非空才启用；失败填 401 并返回 false。
bool authorized(const ServingConfig& cfg, const httplib::Request& req, httplib::Response& res) {
    if (cfg.api_keys.empty()) return true;
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        write_error(res, 401, "UNAUTHORIZED", "missing Authorization header");
        return false;
    }
    const std::string& hdr = it->second;
    constexpr char kPrefix[] = "Bearer ";
    if (hdr.compare(0, sizeof(kPrefix) - 1, kPrefix) != 0 || hdr.size() <= sizeof(kPrefix) - 1) {
        write_error(res, 401, "UNAUTHORIZED",
                    "invalid Authorization header (expect 'Bearer <key>')");
        return false;
    }
    const std::string token = hdr.substr(sizeof(kPrefix) - 1);
    for (const auto& k : cfg.api_keys) {
        if (ct_equal(token, k)) return true;
    }
    write_error(res, 401, "UNAUTHORIZED", "invalid API key");
    return false;
}

}  // namespace

ServingServer::ServingServer(const ServingConfig& cfg, HandleBuilder builder, std::string* err)
    : cfg_(cfg),
      drain_(std::make_shared<DrainState>()),
      repo_(std::make_shared<ModelRepo>(cfg, std::move(builder), err)) {}

ServingServer::~ServingServer() { stop(); }

bool ServingServer::start(std::string* err) {
    if (started_.exchange(true)) {
        if (err) *err = "ServingServer already started";
        return false;
    }
    srv_ = std::make_shared<httplib::Server>();
    if (cfg_.http_threads > 0) {
        srv_->new_task_queue = [n = cfg_.http_threads]() {
            return new httplib::ThreadPool(n, n, 0, 0);
        };
    }
    srv_->set_payload_max_length(cfg_.max_body_bytes);
    register_routes();

    repo_->scan();  // 首次扫描

    const int p = srv_->bind_to_any_port(cfg_.host);
    if (p < 0) {
        if (err) *err = "ServingServer: bind_to_any_port failed on " + cfg_.host;
        started_.store(false);
        return false;
    }
    bound_port_ = p;
    listening_.store(true);
    listen_thread_ = std::thread([this]() {
        srv_->listen_after_bind();
        listening_.store(false);
    });
    srv_->wait_until_ready();
    return true;
}

void ServingServer::stop() {
    if (!started_.exchange(false)) return;  // 幂等
    listening_.store(false);
    auto srv = srv_;
    if (srv) srv->stop();  // 关闭监听 socket → listen_after_bind() 返回
    if (listen_thread_.joinable()) listen_thread_.join();  // join 线程池：同步 handler 于此处完成
    wait_drained();        // 守候游离 fire-and-forget 推理线程跑完（受 bound 超时）
    srv_.reset();
}

bool ServingServer::is_listening() const { return listening_.load(); }
int ServingServer::port() const { return bound_port_; }
ModelRepo* ServingServer::repo() { return repo_.get(); }

void ServingServer::begin_request() {
    auto d = drain_;
    std::lock_guard<std::mutex> lk(d->m);
    ++d->in_flight;
}
void ServingServer::wait_drained() {
    auto d = drain_;
    const auto deadline =
        std::chrono::steady_clock::now() + cfg_.request_timeout + std::chrono::seconds(1);
    std::unique_lock<std::mutex> lk(d->m);
    d->cv.wait_until(lk, deadline, [&d] { return d->in_flight == 0; });
}

void ServingServer::register_routes() {
    auto srv = srv_;

    srv->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const auto models = repo_->list();
        const bool any_ready =
            std::any_of(models.begin(), models.end(), [](const ModelHandle& h) { return h.ready; });
        if (any_ready) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\":\"unavailable\"}", "application/json");
        }
    });

    srv->Get("/readyz", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const auto models = repo_->list();
        const bool all_ready =
            std::all_of(models.begin(), models.end(), [](const ModelHandle& h) { return h.ready; });
        if (all_ready) {
            res.set_content("{\"status\":\"ready\"}", "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\":\"not_ready\"}", "application/json");
        }
    });

    srv->Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        // Task 4 占位；完整 Prometheus 指标在 Task 5 填充。
        res.set_content("# serving metrics (Task 5)\n", "text/plain; version=0.0.4");
    });

    srv->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const auto models = repo_->list();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& h : models)
            arr.push_back({{"name", h.name}, {"version", h.version}, {"ready", h.ready}});
        res.set_content(nlohmann::json{{"models", arr}}.dump(), "application/json");
    });

    srv->Get("/v1/models/:name", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const std::string name = req.path_params.at("name");
        ModelHandle h;
        if (!repo_->get(name, "latest", &h)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + name);
            return;
        }
        res.set_content(nlohmann::json{{"model", {{"name", h.name}, {"version", h.version},
                                                 {"ready", h.ready}}}}
                            .dump(),
                        "application/json");
    });

    srv->Post("/v1/models/:name/infer", [this](const httplib::Request& req,
                                               httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const std::string name = req.path_params.at("name");
        ModelHandle h;
        if (!repo_->get(name, "latest", &h)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + name);
            return;
        }
        if (!h.ready) {
            write_error(res, 503, "MODEL_NOT_READY", "model not ready: " + name);
            return;
        }
        nlohmann::json in;
        try {
            in = nlohmann::json::parse(req.body);
        } catch (...) {
            write_error(res, 400, "BAD_REQUEST", "invalid JSON body");
            return;
        }
        if (!in.is_object()) {
            write_error(res, 400, "BAD_REQUEST", "body must be a JSON object");
            return;
        }

        struct InferJob {
            ModelHandle handle;
            nlohmann::json in;
            nlohmann::json out;
            std::string err;
            bool ok = false;
        };

        // 推理放到游离 worker 线程：文件句柄持有自己的 AsyncModel（shared_ptr），
        // 故不阻塞 httplib 工作线程，且可用 wait_for 施加 cfg_.request_timeout。
        // 线程 lambda 捕获 drain_ 的 shared_ptr（以及持有 AsyncModel 的 job->handle），
        // 不捕获裸 this —— 即使 ServingServer 析构，线程仍借 DrainState/job 存活，无 UAF。
        begin_request();
        auto job = std::make_shared<InferJob>();
        job->handle = std::move(h);
        job->in = std::move(in);
        auto drain = drain_;
        std::promise<void> done;
        auto df = done.get_future();
        std::thread([job, done = std::move(done), drain]() mutable {
            try {
                job->ok = job->handle.infer(job->in, &job->out, &job->err);
            } catch (...) {
                // 无论抛什么异常，都走失败分支继续，避免 in_flight 泄漏与 std::terminate。
                job->ok = false;
                if (job->err.empty()) job->err = "inference threw";
            }
            done.set_value();  // 先解除 httplib worker 阻塞（非超时路径取结果）
            {
                std::lock_guard<std::mutex> lk(drain->m);
                if (drain->in_flight > 0) --drain->in_flight;
            }
            drain->cv.notify_all();  // 在途计数在真正完成时回收（含超时后的后台继续）
        }).detach();

        if (df.wait_for(cfg_.request_timeout) == std::future_status::timeout) {
            // 超时 → 504，后台线程继续跑完（fire-and-forget），不阻塞、不泄漏。
            write_error(res, 504, "TIMEOUT",
                        "inference timed out after " + std::to_string(cfg_.request_timeout.count()) +
                            "ms");
        } else {
            df.get();
            if (job->ok) {
                res.status = 200;
                res.set_content(job->out.dump(), "application/json");
            } else {
                write_error(res, 400, "BAD_REQUEST",
                            job->err.empty() ? "inference failed" : job->err);
            }
        }
    });
}

}  // namespace modeldeploy::serving

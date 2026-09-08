//
// ServingServer —— HTTP 推理网关实现（Task 4 骨架）。
//
// 路由/错误体/Bearer 鉴权/超时/健康/优雅停机见 server.h 注释。
// httplib 工作线程池：默认硬件并发（或 cfg_.http_threads）；见 httplib.h。
//
#include "serving/server.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "core/md_log.h"
#include "httplib.h"

namespace modeldeploy::serving {

// 全局令牌桶（线程安全）。rate<=0（不限流）时恒放行；否则按恒定速率补充令牌，容量=1，
// 故严格按 qps 收敛，无突发窗口——便于测试稳定（qps=1 时并发第 2 个必 429）。
// 定义于此命名空间域，与 server.h 的前置声明（struct TokenBucket;）对应。
class TokenBucket {
public:
    explicit TokenBucket(double qps)
        : rate_(qps), tokens_(1.0), last_(std::chrono::steady_clock::now()) {}
    bool try_acquire() {
        if (rate_ <= 0.0) return true;
        std::lock_guard<std::mutex> lk(m_);
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        tokens_ = std::min(1.0, tokens_ + elapsed * rate_);
        if (tokens_ >= 1.0) {
            tokens_ -= 1.0;
            return true;
        }
        return false;
    }

private:
    std::mutex m_;
    double rate_;
    double tokens_;
    std::chrono::steady_clock::time_point last_;
};

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

// 模型状态 → 前端字符串。
std::string status_str(ModelStatus s) {
    switch (s) {
        case ModelStatus::Unloaded: return "unloaded";
        case ModelStatus::Loading: return "loading";
        case ModelStatus::Ready: return "ready";
        case ModelStatus::Failed: return "failed";
    }
    return "unloaded";
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

// CORS：enable_cors 且请求带 Origin 时，在其上追加允许跨域响应头。OPTIONS 预检由
// register_routes 的 Options("/.*") 处理，且经 post_routing 统一追加（本函数）。
void apply_cors(const ServingConfig& cfg, const httplib::Request& req, httplib::Response& res) {
    if (!cfg.enable_cors) return;
    if (req.headers.find("Origin") == req.headers.end()) return;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Authorization,Content-Type");
}

// 全局令牌桶实现移动到命名空间域（见上方），匿名命名空间仅保留自由函数与局部结构。

// 记录一次 infer 请求的最终状态码到 /metrics（RAII：handler 返回时记录）。
// res 为 httplib handler 的形参，存活期覆盖本 recorder（它与 handler 同作用域）。const
// 引用仅读取 res.status，不修改连接。
struct InferRecorder {
    ServingMetrics* metrics;
    std::string model;
    const httplib::Response& res;
    ~InferRecorder() {
        if (metrics) metrics->record_request(model, res.status);
    }
};

}  // namespace

void ServingMetrics::record_request(const std::string& model, int code) {
    std::lock_guard<std::mutex> lk(m);
    ++requests[model][code];
}

void ServingMetrics::record_inference(const std::string& model, double ms) {
    std::lock_guard<std::mutex> lk(m);
    inference_ms[model].push_back(ms);
}

std::string ServingMetrics::render() const {
    std::lock_guard<std::mutex> lk(m);
    std::string out;
    out += "# HELP modeldeploy_serving_requests_total Number of inference requests served per "
           "model and status code.\n";
    out += "# TYPE modeldeploy_serving_requests_total counter\n";
    for (const auto& [model, codes] : requests) {
        for (const auto& [code, count] : codes) {
            out += "modeldeploy_serving_requests_total{model=\"" + model +
                   "\",code=\"" + std::to_string(code) + "\"} " + std::to_string(count) + "\n";
        }
    }

    out += "# HELP modeldeploy_serving_inference_ms Inference latency in milliseconds.\n";
    out += "# TYPE modeldeploy_serving_inference_ms summary\n";
    for (const auto& [model, samples] : inference_ms) {
        if (samples.empty()) continue;
        double sum = 0.0;
        for (double s : samples) sum += s;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&](double q) {
            if (sorted.empty()) return 0.0;
            const size_t idx =
                static_cast<size_t>(std::ceil(q * static_cast<double>(sorted.size()))) - 1;
            return sorted[std::min(idx, sorted.size() - 1)];
        };
        out += "modeldeploy_serving_inference_ms_sum{model=\"" + model + "\"} " +
               std::to_string(sum) + "\n";
        out += "modeldeploy_serving_inference_ms_count{model=\"" + model + "\"} " +
               std::to_string(samples.size()) + "\n";
        out += "modeldeploy_serving_inference_ms{model=\"" + model +
               "\",quantile=\"0.5\"} " + std::to_string(pct(0.5)) + "\n";
        out += "modeldeploy_serving_inference_ms{model=\"" + model +
               "\",quantile=\"0.95\"} " + std::to_string(pct(0.95)) + "\n";
        out += "modeldeploy_serving_inference_ms{model=\"" + model +
               "\",quantile=\"1\"} " + std::to_string(pct(1.0)) + "\n";
    }
    return out;
}

ServingServer::ServingServer(const ServingConfig& cfg, HandleBuilder builder, std::string* err)
    : cfg_(cfg),
      drain_(std::make_shared<DrainState>()),
      metrics_(std::make_shared<ServingMetrics>()),
      limiter_(std::make_unique<TokenBucket>(cfg.rate_limit_qps)),
      repo_(std::make_shared<ModelRepo>(cfg, std::move(builder), err)) {}

ServingServer::~ServingServer() { stop(); }

bool ServingServer::start(std::string* err) {
    if (started_.exchange(true)) {
        if (err) *err = "ServingServer already started";
        return false;
    }
    srv_ = make_http_server();
    if (cfg_.http_threads > 0) {
        srv_->new_task_queue = [n = cfg_.http_threads]() {
            return new httplib::ThreadPool(n, n, 0, 0);
        };
    }
    srv_->set_payload_max_length(cfg_.max_body_bytes);
    if (!cfg_.web_root.empty()) {
        // 同源静态托管：web_root 非空即在 "/" 挂载静态目录（index.html/MIME/路径穿越
        // 由 httplib 处理）。挂载点下不存在的文件返回 404，不遮蔽已注册的 API 路由。
        // 目录无效仅告警，不致命。
        if (!srv_->set_mount_point("/", cfg_.web_root)) {
            MD_LOG_WARN << "ServingServer: web_root invalid or not a directory: "
                        << cfg_.web_root << std::endl;
        }
    }
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

// 依配置装配 HTTP 或 HTTPS（TLS）服务器。TLS 仅当构建带 OpenSSL（BUILD_SERVING_TLS）
// 且 cfg.enable_tls && tls_cert 非空时启用；否则回退普通 HTTP。TTLS 分支在
// #if defined(MODELDEPLOY_SERVING_TLS) 下编译，本机（无 OpenSSL）不参与。
std::shared_ptr<httplib::Server> ServingServer::make_http_server() {
#if defined(MODELDEPLOY_SERVING_TLS)
    if (cfg_.enable_tls && !cfg_.tls_cert.empty()) {
        return std::make_shared<httplib::SSLServer>(cfg_.tls_cert.c_str(),
                                                    cfg_.tls_key.empty() ? nullptr
                                                                         : cfg_.tls_key.c_str());
    }
#endif
    return std::make_shared<httplib::Server>();
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

    // CORS：enable_cors 时对所有响应（含错误与 OPTIONS 预检）追加跨域头；无 Origin 不加。
    srv->set_post_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        apply_cors(cfg_, req, res);
    });
    srv->Options(
        "/.*", [this](const httplib::Request& req, httplib::Response& res) {
            if (!cfg_.enable_cors) {  // 关闭 CORS 时预检报 404，不泄露跨域许可
                res.status = 404;
                return;
            }
            res.status = 204;  // 跨域头由 post_routing 统一追加
        });

    srv->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const auto models = repo_->list();
        const bool any_ready =
            std::any_of(models.begin(), models.end(),
                        [](const ModelHandle& h) { return h.status == ModelStatus::Ready; });
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
        // 单槽语义：有模型在后台构建 → 瞬态 loading；否则仅当恰有一个活跃 Ready 才 ready。
        const bool any_loading =
            std::any_of(models.begin(), models.end(),
                        [](const ModelHandle& h) { return h.status == ModelStatus::Loading; });
        if (any_loading) {
            res.status = 503;
            res.set_content("{\"status\":\"loading\"}", "application/json");
            return;
        }
        int ready_count = 0;
        for (const auto& h : models)
            if (h.status == ModelStatus::Ready) ++ready_count;
        if (ready_count == 1) {
            res.set_content("{\"status\":\"ready\"}", "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\":\"not_ready\"}", "application/json");
        }
    });

    srv->Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        // Prometheus 文本（prometheus 客户端标准 text format 0.0.4）。
        res.set_content(metrics_->render(), "text/plain; version=0.0.4");
    });

    srv->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const auto models = repo_->list();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& h : models)
            arr.push_back({{"id", h.name}, {"name", h.name}, {"display", h.display},
                           {"version", h.version}, {"type", h.type}, {"labels", h.labels},
                           {"input_size", h.input_size}, {"status", status_str(h.status)},
                           {"ready", h.status == ModelStatus::Ready}, {"error", h.error}});
        res.set_content(nlohmann::json{{"models", arr}}.dump(), "application/json");
    });

    srv->Get("/v1/models/:name", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const std::string name = req.path_params.at("name");
        ModelHandle h;
        if (!repo_->get(name, &h)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + name);
            return;
        }
        res.set_content(nlohmann::json{{"model", {{"id", h.name}, {"name", h.name},
                                                 {"display", h.display}, {"version", h.version},
                                                 {"type", h.type}, {"labels", h.labels},
                                                 {"input_size", h.input_size},
                                                 {"status", status_str(h.status)},
                                                 {"ready", h.status == ModelStatus::Ready},
                                                 {"error", h.error}}}}
                            .dump(),
                        "application/json");
    });

    srv->Post("/v1/models/:id/load", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const std::string id = req.path_params.at("id");
        ModelHandle probe;
        if (!repo_->get(id, &probe)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + id);
            return;
        }
        // 后台线程调 load，避免阻塞 HTTP worker；立即返回当前目录状态。并发保护由
        // ModelRepo 的 Loading 在途状态承担（同一 id 重复 load 为 no-op）。
        auto repo = repo_;
        std::thread([repo, id]() { std::string err; repo->load(id, &err); }).detach();
        ModelHandle h;
        repo_->get(id, &h);
        res.set_content(nlohmann::json{{"id", h.name}, {"status", status_str(h.status)}}.dump(),
                        "application/json");
    });

    srv->Post("/v1/models/:id/unload", [this](const httplib::Request& req, httplib::Response& res) {
        if (!authorized(cfg_, req, res)) return;
        const std::string id = req.path_params.at("id");
        ModelHandle h;
        if (!repo_->get(id, &h)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + id);
            return;
        }
        repo_->unload(id);
        repo_->get(id, &h);
        res.set_content(nlohmann::json{{"id", h.name}, {"status", status_str(h.status)}}.dump(),
                        "application/json");
    });

    srv->Post("/v1/models/:name/infer", [this](const httplib::Request& req,
                                               httplib::Response& res) {
        const std::string name = req.path_params.at("name");
        // RAII：handler 返回时把最终状态码记入 /metrics（覆盖 429/401/404/503/400/200/504）。
        InferRecorder recorder{metrics_.get(), name, res};
        if (!authorized(cfg_, req, res)) return;
        ModelHandle h;
        if (!repo_->get(name, &h)) {
            write_error(res, 404, "MODEL_NOT_FOUND", "model not found: " + name);
            return;
        }
        // Failed 模型不重建：直接报错，避免每次 /infer 都重建重型模型。
        if (h.status == ModelStatus::Failed) {
            write_error(res, 503, "MODEL_NOT_READY", "model failed to load: " + h.error);
            return;
        }
        // 懒加载：非 Ready 且非 Loading 时同步 load 一次；此后仍非 Ready → 不重试，直接报错。
        if (h.status != ModelStatus::Ready && h.status != ModelStatus::Loading) {
            std::string load_err;
            repo_->load(name, &load_err);
            repo_->get(name, &h);
        }
        if (h.status != ModelStatus::Ready) {
            write_error(res, 503, "MODEL_NOT_READY",
                        h.status == ModelStatus::Failed ? "model failed to load: " + h.error
                                                        : "model not ready: " + name);
            return;
        }
        // 限流（全局令牌桶）：超限 429，不入 in_flight、不占推理。
        if (!limiter_->try_acquire()) {
            write_error(res, 429, "RATE_LIMITED", "rate limit exceeded");
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

        // 推理耗时记入 /metrics（含超时样本：观测时长为 request_timeout）。
        const auto t0 = std::chrono::steady_clock::now();

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
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
            metrics_->record_inference(name, elapsed_ms);
            write_error(res, 504, "TIMEOUT",
                        "inference timed out after " + std::to_string(cfg_.request_timeout.count()) +
                            "ms");
        } else {
            df.get();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
            metrics_->record_inference(name, elapsed_ms);
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

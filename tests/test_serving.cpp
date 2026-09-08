#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#include "serving/config.h"
#include "serving/model_repo.h"
#include "serving/manifest.h"
#include "serving/model_entry.h"
#include "serving/server.h"
#include "serving/adapters.h"
#include "pipeline/async_model.h"
#include "vision/common/image_data.h"
#include "vision/common/result_json.h"
#include "../third_party/httplib.h"

namespace fs = std::filesystem;
using namespace modeldeploy::serving;

namespace {

// 内嵌 1×1 有效 PNG 的 base64（宽高已知：1×1）。来源：标准 1×1 透明 PNG。
const std::string PNG1X1_B64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

// 测试自包含 base64 解码（SDK 的 utils::base64_decode 未导出，测试侧不复用）。
std::vector<unsigned char> b64_decode(const std::string& s) {
    static const std::string tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> out;
    int val = 0;
    int bits = -8;
    for (unsigned char c : s) {
        if (c == '=') break;
        auto pos = tbl.find(static_cast<char>(c));
        if (pos == std::string::npos) break;
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<unsigned char>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

// 最小 FakeModel：R = std::string，predict/batch_predict 返回 w/h（与异步契约匹配）。
// delay 可配：>0 时 infer 前 sleep，用于 504 超时 / 在途停机用例。
// started/finished 为可选事件标志（均为 shared_ptr，供测试事件同步）：
//   started  置位于 infer（predict/batch_predict）真正开始前——确认请求已在途；
//   finished 置位于 infer 真正结束后——确认游离线程已安全跑完（可用于析构后守候）。
struct FakeModel {
    using result_type = std::string;
    int predict_calls = 0;
    std::chrono::milliseconds delay{0};
    std::shared_ptr<std::atomic<bool>> started;
    std::shared_ptr<std::atomic<bool>> finished;
    static std::string big_result(int w, int h) {
        return "w=" + std::to_string(w) + ",h=" + std::to_string(h);
    }
    void mark_started() {
        if (started) started->store(true);
    }
    void mark_finished() {
        if (finished) finished->store(true);
    }
    bool predict(const modeldeploy::vision::ImageData& img, std::string* out,
                 TimerArray* /*timers*/ = nullptr) {
        ++predict_calls;
        mark_started();
        std::this_thread::sleep_for(delay);
        *out = big_result(img.width(), img.height());
        mark_finished();
        return true;
    }
    bool batch_predict(const std::vector<modeldeploy::vision::ImageData>& imgs,
                       std::vector<std::string>* outs,
                       TimerArray* /*timers*/ = nullptr) {
        mark_started();
        outs->clear();
        for (auto& im : imgs) {
            std::this_thread::sleep_for(delay);
            outs->push_back(big_result(im.width(), im.height()));
        }
        mark_finished();
        return true;
    }
};

// StubDet —— 模拟真实 det 模型的结果形态（result_type=std::vector<DetectionResult>）。
// 不依赖真实权重；predict/batch_predict 各产出一个框，用于 typed 结果 JSON round-trip。
struct StubDet {
    using result_type = std::vector<modeldeploy::vision::DetectionResult>;
    static result_type make_one(float x, float y, float w, float h, int label, float score) {
        result_type v(1);
        v[0].box = {x, y, w, h};
        v[0].label_id = label;
        v[0].score = score;
        return v;
    }
    bool predict(const modeldeploy::vision::ImageData&, result_type* r, TimerArray* = nullptr) {
        *r = make_one(10.f, 20.f, 100.f, 50.f, 0, 0.9f);
        return true;
    }
    bool batch_predict(const std::vector<modeldeploy::vision::ImageData>& imgs,
                       std::vector<result_type>* rs, TimerArray* = nullptr) {
        rs->clear();
        for (size_t i = 0; i < imgs.size(); ++i)
            rs->push_back(make_one(10.f, 20.f, 100.f, 50.f, 0, 0.9f));
        return true;
    }
};

}  // namespace

TEST_CASE("typed result JSON via make_model_handle (CPU)", "[serving]") {
    auto h = make_model_handle<StubDet>("det", std::make_unique<StubDet>());
    REQUIRE(h.status == ModelStatus::Ready);
    nlohmann::json out;
    std::string err;
    REQUIRE(h.infer(nlohmann::json{{"image", PNG1X1_B64}}, &out, &err));
    REQUIRE(out["model"] == "det");
    REQUIRE(out.contains("results"));
    REQUIRE(out["results"].is_array());
    REQUIRE(out["results"].size() == 1);
    REQUIRE(out["results"][0]["score"] == Catch::Approx(0.9f));
    REQUIRE(out["results"][0]["label_id"] == 0);
    REQUIRE(out["results"][0]["box"]["x"] == Catch::Approx(10.0f));
    REQUIRE(out["results"][0]["box"]["width"] == Catch::Approx(100.0f));
    REQUIRE(out.contains("duration_ms"));
}

namespace {

// 返回唯一临时目录（不放在仓库/源目录）；调用方可 remove_all 清理。
std::string make_temp_repo() {
    static std::mt19937 rng{std::random_device{}()};
    auto base = fs::temp_directory_path() /
                ("md_serving_" + std::to_string(static_cast<unsigned>(rng())));
    fs::create_directories(base);
    return base.string();
}

// 测试用的 manifest 条目描述。
struct TestModel {
    std::string id;
    std::string type = "det";
    std::string display;
    std::vector<int> input_size;
    std::vector<std::string> labels;
};

// 在 dir 下写 manifest.json（base=dir），返回其路径。BASE 指向 dir，目录不存在则创建。
std::string write_manifest(const std::string& dir, const std::vector<TestModel>& mods) {
    fs::create_directories(dir);
    nlohmann::json j;
    j["base"] = dir;
    j["models"] = nlohmann::json::array();
    for (const auto& m : mods) {
        nlohmann::json e;
        e["id"] = m.id;
        e["type"] = m.type;
        if (!m.display.empty()) e["display"] = m.display;
        if (!m.input_size.empty()) e["input_size"] = m.input_size;
        if (!m.labels.empty()) e["labels"] = m.labels;
        j["models"].push_back(e);
    }
    auto path = (fs::path(dir) / "manifest.json").string();
    { std::ofstream f(path); f << j.dump(); }
    return path;
}

// 注入假 InferFn：调用时把 manifest id 写进 out["handle"]，用于核对句柄身份。
HandleBuilder echo_builder() {
    return [](const ManifestModel& m, const std::string&) -> ModelHandle {
        std::string key = m.id;
        ModelHandle h;
        h.name = m.id;
        h.infer = InferFn([key](const nlohmann::json& in, nlohmann::json* out, std::string* err) {
            (void)in;
            (void)err;
            if (out) (*out)["handle"] = key;
            return true;
        });
        return h;
    };
}

std::string handle_of(const ModelHandle& h) {
    nlohmann::json out;
    std::string err;
    REQUIRE(h.infer(nlohmann::json::object(), &out, &err));
    REQUIRE(out.contains("handle"));
    return out["handle"].get<std::string>();
}

// 写一个临时 web_root：index.html + app.js + 一个嵌套目录。
std::string make_web_root() {
    static std::mt19937 rng{std::random_device{}()};
    auto r = fs::temp_directory_path() /
             ("md_web_" + std::to_string(static_cast<unsigned>(rng())));
    fs::create_directories(r / "assets");
    { std::ofstream f(r / "index.html"); f << "<h1>ok</h1>"; }
    { std::ofstream f(r / "app.js"); f << "console.log('x')"; }
    { std::ofstream f(r / "assets" / "logo.svg"); f << "<svg/>"; }
    return r.string();
}

}  // namespace

TEST_CASE("ModelRepo scan get list", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det", "Det"}, {"cls", "cls", "Cls"}});

    ModelRepo m(cfg, echo_builder());
    auto changed = m.scan();
    REQUIRE(std::find(changed.begin(), changed.end(), "det") != changed.end());
    REQUIRE(std::find(changed.begin(), changed.end(), "cls") != changed.end());

    ModelHandle h;
    REQUIRE(m.get("det", &h));
    REQUIRE(h.name == "det");
    REQUIRE(h.display == "Det");
    REQUIRE(h.version == "1");
    REQUIRE(h.type == "det");
    REQUIRE(h.status == ModelStatus::Unloaded);
    REQUIRE_FALSE(h.infer);  // 目录只含元数据，不实例化

    std::string err;
    REQUIRE(m.load("det", &err));
    REQUIRE(err.empty());
    REQUIRE(m.get("det", &h));
    REQUIRE(h.status == ModelStatus::Ready);
    REQUIRE(handle_of(h) == "det");  // echo builder 按 manifest id 回显

    REQUIRE(m.get("cls", &h));
    REQUIRE(h.status == ModelStatus::Unloaded);

    auto all = m.list();
    REQUIRE(all.size() == 2);
    std::vector<std::string> names;
    for (auto& e : all) names.push_back(e.name);
    REQUIRE(std::find(names.begin(), names.end(), "det") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "cls") != names.end());

    // 单槽：load cls 卸下 det。
    REQUIRE(m.load("cls", &err));
    REQUIRE(m.get("det", &h));
    REQUIRE(h.status == ModelStatus::Unloaded);
    REQUIRE(m.get("cls", &h));
    REQUIRE(h.status == ModelStatus::Ready);

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo manifest order & defaults", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"a", "det"}, {"b", "cls"}});

    ModelRepo m(cfg, echo_builder());
    m.scan();

    auto all = m.list();
    REQUIRE(all.size() == 2);
    REQUIRE(all[0].name == "a");   // 目录序 = manifest 序
    REQUIRE(all[1].name == "b");
    REQUIRE(all[0].input_size == std::vector<int>{640, 640});  // 缺省 640x640
    ModelHandle h;
    REQUIRE(m.get("a", &h));
    REQUIRE(h.version == "1");  // 恒 "1"

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo hot update (manifest rescan)", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ModelRepo m(cfg, echo_builder());
    m.scan();
    std::string err;
    REQUIRE(m.load("det", &err));
    ModelHandle oldh;
    REQUIRE(m.get("det", &oldh));
    REQUIRE(oldh.status == ModelStatus::Ready);
    auto old_infer = oldh.infer;  // 值拷贝：旧持有者可独立于 repo 完成

    // 重写 manifest 增加 cls；rescan 新增它，同时 det 回落目录态。
    write_manifest(repo, {{"det", "det"}, {"cls", "cls"}});
    auto changed = m.scan();
    REQUIRE(std::find(changed.begin(), changed.end(), "cls") != changed.end());
    REQUIRE(std::find(changed.begin(), changed.end(), "det") == changed.end());  // 非新增

    ModelHandle h;
    REQUIRE(m.get("cls", &h));
    REQUIRE(h.status == ModelStatus::Unloaded);
    REQUIRE(m.get("det", &h));
    REQUIRE(h.status == ModelStatus::Unloaded);  // 重扫回落 Unloaded

    nlohmann::json out;
    REQUIRE(old_infer(nlohmann::json::object(), &out, nullptr));
    REQUIRE(out["handle"] == "det");

    fs::remove_all(repo);
}

TEST_CASE("ModelRepo unknown names", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ModelRepo m(cfg);
    m.scan();

    ModelHandle h;
    REQUIRE_FALSE(m.get("nope", &h));
    std::string err;
    REQUIRE_FALSE(m.load("nope", &err));

    fs::remove_all(repo);
}

TEST_CASE("ModelEntry image_path & image infer + reuse", "[serving]") {
    // base64 → 字节 → 写临时 PNG（1×1）
    auto bytes = b64_decode(PNG1X1_B64);
    REQUIRE(bytes.size() > 0);

    auto tmp = fs::temp_directory_path() /
               ("md_entry_" + std::to_string(static_cast<unsigned>(std::random_device{}())));
    fs::create_directories(tmp);
    fs::path png = tmp / "one.png";
    {
        std::ofstream ofs(png, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }

    auto handle =
        make_model_handle<FakeModel>("fake", std::make_unique<FakeModel>());
    REQUIRE(handle.name == "fake");
    REQUIRE(handle.status == ModelStatus::Ready);

    nlohmann::json out;
    std::string err;

    // 1) image_path 路径
    REQUIRE(handle.infer(nlohmann::json{{"image_path", png.string()}}, &out, &err));
    REQUIRE(out["results"] == "w=1,h=1");
    REQUIRE(out["model"] == "fake");
    REQUIRE(out["duration_ms"].is_number());

    // 2) image(base64) 路径
    REQUIRE(handle.infer(nlohmann::json{{"image", PNG1X1_B64}}, &out, &err));
    REQUIRE(out["results"] == "w=1,h=1");
    REQUIRE(out["model"] == "fake");

    // 3) 复用：同一 handle 多次 infer（AsyncModel 单实例可用）
    REQUIRE(handle.infer(nlohmann::json{{"image", PNG1X1_B64}}, &out, &err));
    REQUIRE(out["results"] == "w=1,h=1");
    REQUIRE(handle.infer(nlohmann::json{{"image_path", png.string()}}, &out, &err));
    REQUIRE(out["results"] == "w=1,h=1");

    fs::remove_all(tmp);
}

TEST_CASE("ModelEntry bad input", "[serving]") {
    auto handle =
        make_model_handle<FakeModel>("fake", std::make_unique<FakeModel>());
    nlohmann::json out;
    std::string err;

    // 无 image/image_path → false + err
    REQUIRE_FALSE(handle.infer(nlohmann::json::object(), &out, &err));
    REQUIRE_FALSE(err.empty());

    // 非法 base64（非 base64 字符）→ false + err
    err.clear();
    REQUIRE_FALSE(handle.infer(nlohmann::json{{"image", "%%%非法%%%"}}, &out, &err));
    REQUIRE_FALSE(err.empty());

    // 不存在的 image_path → false + err
    err.clear();
    REQUIRE_FALSE(handle.infer(
        nlohmann::json{{"image_path", fs::temp_directory_path() / "no_such_file.png"}}, &out, &err));
    REQUIRE_FALSE(err.empty());
}

namespace {

// 注入真实 AsyncModel（FakeModel）的 HandleBuilder：make_model_handle 启动失败时兜底为失败 InferFn。
HandleBuilder fake_model_builder() {
    return [](const ManifestModel& m, const std::string&) -> ModelHandle {
        ModelHandle h;
        try {
            auto mh = make_model_handle<FakeModel>(m.id, std::make_unique<FakeModel>());
            h = mh;
        } catch (...) {
            h.infer = InferFn([](const nlohmann::json&, nlohmann::json*, std::string* err) {
                if (err) *err = "fake model start failed";
                return false;
            });
        }
        return h;
    };
}

// 带事件标志的慢模型 HandleBuilder：started/finished 与 FakeModel 对应字段共享，
// 供测试以事件同步代替固定 sleep，确认推理确已在途（started）与确已跑完（finished）。
HandleBuilder slow_model_builder_flag(std::chrono::milliseconds delay,
                                      std::shared_ptr<std::atomic<bool>> started,
                                      std::shared_ptr<std::atomic<bool>> finished) {
    return [delay, started, finished](const ManifestModel& m, const std::string&) -> ModelHandle {
        ModelHandle h;
        try {
            auto model = std::make_unique<FakeModel>();
            model->delay = delay;
            model->started = started;
            model->finished = finished;
            h = make_model_handle<FakeModel>(m.id, std::move(model));
        } catch (...) {
            h.infer = InferFn([](const nlohmann::json&, nlohmann::json*, std::string* err) {
                if (err) *err = "fake model start failed";
                return false;
            });
        }
        return h;
    };
}

// 慢模型 HandleBuilder：每次 infer 前 sleep delay，用于 504 超时 / 在途停机用例。
HandleBuilder slow_model_builder(std::chrono::milliseconds delay) {
    return slow_model_builder_flag(delay, nullptr, nullptr);
}

// 事件同步：阻塞直到 pred 返回 true（避免在压测下用固定 sleep 猜时序的 flake）。
void wait_until(const std::function<bool()>& pred) {
    while (!pred()) std::this_thread::yield();
}

// 起服并断言随机端口；返回监听中的随机端口供 httplib::Client 使用。
int start_listening(ServingServer& srv) {
    REQUIRE(srv.start());
    REQUIRE(srv.is_listening());
    int port = srv.port();
    REQUIRE(port > 0);
    return port;
}

httplib::Client make_client(int port) { return httplib::Client("127.0.0.1", port); }

// 统一校验错误体：{ "error": { "code": ..., "message": ... } }
void require_error(const httplib::Result& res, int status, const std::string& code) {
    REQUIRE(res);
    REQUIRE(res->status == status);
    auto j = nlohmann::json::parse(res->body);
    REQUIRE(j.contains("error"));
    REQUIRE(j["error"]["code"].get<std::string>() == code);
}

// 起服后显式懒加载指定模型（懒加载语义：启动不实例化，需用前 load）。
void preload(ServingServer& srv, const std::string& id) {
    std::string err;
    REQUIRE(srv.repo()->load(id, &err));
}

// 返回就绪态 infer 的 builder（元数据由 manifest 提供）。
HandleBuilder meta_builder() {
    return [](const ManifestModel&, const std::string&) -> ModelHandle {
        ModelHandle h;
        h.infer = [](const nlohmann::json&, nlohmann::json* out, std::string*) {
            if (out) (*out)["meta_ok"] = true;
            return true;
        };
        return h;
    };
}

}  // namespace

TEST_CASE("model metadata in /v1/models", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(
        repo, {{"det", "det", "Det", {640, 640}, {"person", "car"}}, {"cls", "cls"}});
    ServingServer srv(cfg, meta_builder());
    std::string err;
    REQUIRE(srv.start(&err));
    httplib::Client cli("127.0.0.1", srv.port());

    auto all0 = cli.Get("/v1/models");
    REQUIRE(all0);
    REQUIRE(all0->status == 200);
    auto j0 = nlohmann::json::parse(all0->body);
    REQUIRE(j0["models"].is_array());
    nlohmann::json det;
    for (auto& m : j0["models"]) if (m["name"] == "det") det = m;
    REQUIRE(!det.is_null());
    REQUIRE(det["type"] == "det");
    REQUIRE(det["labels"].size() == 2);
    REQUIRE(det["labels"][0] == "person");
    REQUIRE(det["input_size"][0] == 640);
    REQUIRE(det["ready"] == false);  // 目录态未实例化

    // 显式 load → ready
    REQUIRE(srv.repo()->load("det", &err));
    auto all = cli.Get("/v1/models");
    REQUIRE(all);
    REQUIRE(all->status == 200);
    auto j = nlohmann::json::parse(all->body);
    for (auto& m : j["models"]) if (m["name"] == "det") det = m;
    REQUIRE(det["ready"] == true);

    auto one = cli.Get("/v1/models/cls");
    REQUIRE(one);
    REQUIRE(one->status == 200);
    auto j1 = nlohmann::json::parse(one->body);
    REQUIRE(j1["model"]["type"] == "cls");

    srv.stop();
    fs::remove_all(repo);
}

// 端到端：起服（随机端口）→ httplib::Client 发请求 → stop() + 清理临时 repo。
TEST_CASE("ServingServer 200 infer", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    REQUIRE(srv.repo() != nullptr);
    preload(srv, "det");

    auto cli = make_client(port);
    auto res = cli.Post("/v1/models/det/infer", nlohmann::json{{"image", PNG1X1_B64}}.dump(),
                        "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = nlohmann::json::parse(res->body);
    REQUIRE(body["results"] == "w=1,h=1");
    REQUIRE(body["model"] == "det");

    srv.stop();
    fs::remove_all(repo);
}

TEST_CASE("ServingServer 404 model not found", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);

    auto cli = make_client(port);
    require_error(cli.Post("/v1/models/nope/infer", "{}", "application/json"), 404,
                  "MODEL_NOT_FOUND");
    require_error(cli.Get("/v1/models/nope"), 404, "MODEL_NOT_FOUND");

    srv.stop();
    fs::remove_all(repo);
}

TEST_CASE("ServingServer 400 bad request", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    require_error(cli.Post("/v1/models/det/infer", "{}", "application/json"), 400, "BAD_REQUEST");

    srv.stop();
    fs::remove_all(repo);
}

TEST_CASE("ServingServer bearer auth", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.api_keys = {"key1"};

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    std::string body = nlohmann::json{{"image", PNG1X1_B64}}.dump();

    // 无 header → 401
    require_error(cli.Post("/v1/models/det/infer", body, "application/json"), 401, "UNAUTHORIZED");
    // 错误 key → 401
    require_error(cli.Post("/v1/models/det/infer", httplib::Headers{{"Authorization", "Bearer wrong"}},
                           body, "application/json"), 401, "UNAUTHORIZED");
    // 正确 key → 200
    auto res = cli.Post("/v1/models/det/infer", httplib::Headers{{"Authorization", "Bearer key1"}},
                        body, "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    srv.stop();
    fs::remove_all(repo);
}

TEST_CASE("ServingServer health & readyz", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    auto h = cli.Get("/health");
    REQUIRE(h);
    REQUIRE(h->status == 200);
    auto r = cli.Get("/readyz");
    REQUIRE(r);
    REQUIRE(r->status == 200);

    srv.stop();
    fs::remove_all(repo);
}

TEST_CASE("ServingServer model list", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}, {"cls", "cls"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);

    auto cli = make_client(port);
    auto res = cli.Get("/v1/models");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = nlohmann::json::parse(res->body);
    REQUIRE(body.contains("models"));
    bool has_det = false;
    for (auto& m : body["models"]) {
        if (m["name"].get<std::string>() == "det") has_det = true;
    }
    REQUIRE(has_det);

    srv.stop();
    fs::remove_all(repo);
}

// 优雅停机：起服后立即 stop()（含注入模型句柄在途场景的干净析构），可重复调用。
TEST_CASE("ServingServer graceful stop", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    REQUIRE(port > 0);

    srv.stop();   // 幂等
    srv.stop();
    REQUIRE_FALSE(srv.is_listening());
    fs::remove_all(repo);
}

// 504 超时：慢模型比 request_timeout 慢得多 → 期望 504 TIMEOUT；随后 stop() 等后台
// 线程跑完后安全返回，无崩溃/无 hang（UAF 修复的回归护栏）。
TEST_CASE("ServingServer 504 timeout", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.request_timeout = std::chrono::milliseconds(50);

    ServingServer srv(cfg, slow_model_builder(std::chrono::milliseconds(500)));
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    auto res = cli.Post("/v1/models/det/infer", nlohmann::json{{"image", PNG1X1_B64}}.dump(),
                        "application/json");
    require_error(res, 504, "TIMEOUT");

    // 超时后后台推理线程仍在跑；stop() 守候其完成（受 deadline 约束），须安全返回。
    srv.stop();
    REQUIRE_FALSE(srv.is_listening());
    fs::remove_all(repo);
}

// 在途时 stop：后台线程发起慢推理，主线程在推理尚未结束时调用 stop()，须安全返回、
// 进程不崩、临时资源清理干净（依赖 drain_ 堆对象保证游离线程不触已析构 this）。
TEST_CASE("ServingServer stop while inference in flight", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.request_timeout = std::chrono::milliseconds(60000);  // 长超时：走非超时完整路径

    auto started = std::make_shared<std::atomic<bool>>(false);
    ServingServer srv(cfg, slow_model_builder_flag(std::chrono::milliseconds(300), started,
                                                   nullptr));
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    bool requester_ok = false;
    // 独立线程发起请求，构造「推理在途」；主线程随即 stop()。
    std::thread requester([&] {
        auto r = cli.Post("/v1/models/det/infer", nlohmann::json{{"image", PNG1X1_B64}}.dump(),
                          "application/json");
        requester_ok = r && r->status == 200;
    });
    // 事件同步：确认请求确已进入推理（started 置位）后再 stop，杜绝固定 sleep 的
    // 「stop 早于请求入队」flake；此时 handler 已 begin_request()，推理线程必在途。
    wait_until([&] { return started->load(); });

    srv.stop();  // 推理在途时停机：须安全返回、等待后台完成、无 crash/hang
    REQUIRE_FALSE(srv.is_listening());
    requester.join();
    REQUIRE(requester_ok);

    fs::remove_all(repo);
}

// drain 超期回归：推理耗时远大于 wait_drained deadline（request_timeout=50ms + 约 1s），
// stop() 的 wait_drained() 必然超期返回；随后 ServingServer 先完成析构，游离推理线程
// 稍后才跑完——验证不崩溃、不 hang、无泄漏（UAF 修复的关键护栏：脱离 this 靠 shared
// DrainState/job 存活）。
TEST_CASE("ServingServer drain deadline exceeded safe dtor (detached thread outlives server)",
          "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.request_timeout = std::chrono::milliseconds(50);  // wait_drained deadline ≈ 50ms + 1s

    // 推理 3s ≫ deadline ≈ 1.05s：wait_drained 必超期返回，且析构先于推理完成。
    auto started = std::make_shared<std::atomic<bool>>(false);
    auto finished = std::make_shared<std::atomic<bool>>(false);

    bool finished_at_stop = false;
    {
        ServingServer srv(cfg, slow_model_builder_flag(std::chrono::milliseconds(3000), started,
                                                       finished));
        int port = start_listening(srv);
        preload(srv, "det");

        auto cli = make_client(port);
        // 独立线程发 POST：handler 在 request_timeout=50ms 内返回（504/或超时），
        // 后台 fire-and-forget 推理线程继续跑（推理在途）。
        std::thread requester([&] {
            cli.Post("/v1/models/det/infer", nlohmann::json{{"image", PNG1X1_B64}}.dump(),
                     "application/json");
        });
        // 事件同步：确认请求确已在途（推理线程已 begin_request + started 置位）。
        wait_until([&] { return started->load(); });

        // stop() → wait_drained 在 deadline（~1.05s）超期返回，而非等到推理（3s）跑完。
        srv.stop();
        finished_at_stop = finished->load();
        // 关键断言：stop() 返回时推理尚未完成 ⇒ wait_drained 确实超期（未真 drain 干净）。
        REQUIRE_FALSE(finished_at_stop);
        REQUIRE_FALSE(srv.is_listening());
        requester.join();
    }
    // 作用域退出 → ServingServer 已析构；游离推理线程仍存活且稍后安全跑完（不碰 this）。

    // 事件同步：守候游离线程在 server 析构后仍安全完成（不崩、不 hang）。
    wait_until([&] { return finished->load(); });

    fs::remove_all(repo);
}

TEST_CASE("ServingServer rate limit 429", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.rate_limit_qps = 1;  // 严格 1 qps，无突发窗口 → 连发第 2 个必 429

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    auto body = nlohmann::json{{"image", PNG1X1_B64}}.dump();
    auto first = cli.Post("/v1/models/det/infer", body, "application/json");
    REQUIRE(first);
    REQUIRE(first->status == 200);  // 首令牌可用

    auto second = cli.Post("/v1/models/det/infer", body, "application/json");
    require_error(second, 429, "RATE_LIMITED");

    srv.stop();
    fs::remove_all(repo);
}

// /metrics Prometheus 文本：计数每模型每状态码、推理耗时聚合（sum/count 与分位数）。
TEST_CASE("ServingServer metrics text", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    auto body = nlohmann::json{{"image", PNG1X1_B64}}.dump();
    REQUIRE(cli.Post("/v1/models/det/infer", body, "application/json")->status == 200);
    REQUIRE(cli.Post("/v1/models/det/infer", body, "application/json")->status == 200);
    REQUIRE(cli.Post("/v1/models/nope/infer", body, "application/json")->status == 404);  // 计入 404

    auto res = cli.Get("/metrics");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    const std::string text = res->body;
    // 计数器：det 的 200 ×2、nope 的 404 ×1
    REQUIRE(text.find("modeldeploy_serving_requests_total{model=\"det\",code=\"200\"} 2") !=
            std::string::npos);
    REQUIRE(text.find("modeldeploy_serving_requests_total{model=\"nope\",code=\"404\"} 1") !=
            std::string::npos);
    // 推理耗时聚合：det 有 2 个样本，sum/count 行存在
    REQUIRE(text.find("modeldeploy_serving_inference_ms_count{model=\"det\"} 2") !=
            std::string::npos);
    REQUIRE(text.find("modeldeploy_serving_inference_ms_sum{model=\"det\"}") !=
            std::string::npos);

    srv.stop();
    fs::remove_all(repo);
}

// CORS：enable_cors（默认 true）时 OPTIONS 预检回带跨域头；关闭后不加。
TEST_CASE("ServingServer cors headers", "[serving]") {
    auto repo = make_temp_repo();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});

    ServingServer srv(cfg, fake_model_builder());
    int port = start_listening(srv);
    preload(srv, "det");

    auto cli = make_client(port);
    auto pre = cli.Options(
        "/v1/models/det/infer",
        httplib::Headers{{"Origin", "https://example.com"}, {"Access-Control-Request-Method", "POST"}});
    REQUIRE(pre);
    REQUIRE(pre->status == 204);
    REQUIRE(pre->get_header_value("Access-Control-Allow-Origin") == "*");
    REQUIRE(pre->get_header_value("Access-Control-Allow-Methods") == "GET,POST,OPTIONS");
    REQUIRE(pre->get_header_value("Access-Control-Allow-Headers") == "Authorization,Content-Type");

    // 带 Origin 的普通请求也会带上跨域头
    auto get = cli.Get("/health", httplib::Headers{{"Origin", "https://example.com"}});
    REQUIRE(get);
    REQUIRE(get->status == 200);
    REQUIRE(get->get_header_value("Access-Control-Allow-Origin") == "*");

    srv.stop();
    fs::remove_all(repo);

    // 关闭 CORS：OPTIONS 预检 404，普通响应也无跨域头。
    ServingConfig cfg2;
    cfg2.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg2.enable_cors = false;
    ServingServer srv2(cfg2, fake_model_builder());
    int port2 = start_listening(srv2);
    preload(srv2, "det");
    httplib::Client cli2("127.0.0.1", port2);
    auto pre2 = cli2.Options("/v1/models/det/infer",
                             httplib::Headers{{"Origin", "https://example.com"}});
    REQUIRE(pre2);
    REQUIRE(pre2->status == 404);
    REQUIRE_FALSE(pre2->has_header("Access-Control-Allow-Origin"));
    auto get2 = cli2.Get("/health", httplib::Headers{{"Origin", "https://example.com"}});
    REQUIRE(get2);
    REQUIRE(get2->status == 200);
    REQUIRE_FALSE(get2->has_header("Access-Control-Allow-Origin"));
    srv2.stop();
    fs::remove_all(repo);
}

// TLS：HTTPS 服务器装配。仅当构建带 OpenSSL（BUILD_SERVING_TLS）时编译；本机无 OpenSSL
// 故该用例不参与编译（标签 [serving-tls]，`[serving]~[serving-tls]` 排除）。
#if defined(MODELDEPLOY_SERVING_TLS)
TEST_CASE("ServingServer tls branch", "[serving][serving-tls]") {
    // 自签证书导入：仅验证 enable_tls + tls_cert 走 SSLServer 装配路径。
    ServingConfig cfg;
    auto dir = make_temp_repo();
    cfg.model_repo = write_manifest(dir, {{"det", "det"}});
    cfg.enable_tls = true;
    // 无合法证书文件时 SSLServer 构造可能失败 → start() 可能 false；此处仅验证
    // 配置了 TLS 且未崩溃地走到 listen（错误在 err 中返回，不抛跨对象）。
    ServingServer srv(cfg, fake_model_builder());
    std::string err;
    bool ok = srv.start(&err);
    srv.stop();
    fs::remove_all(dir);
    // 本用例仅验证 TLS 装配分支存在且可安全起停；不要求无证书时也能成功监听。
    (void)ok;
    (void)err;
}
#endif

TEST_CASE("serving static hosting same-origin", "[serving]") {
    auto repo = make_temp_repo();
    auto web = make_web_root();
    ServingConfig cfg;
    cfg.model_repo = write_manifest(repo, {{"det", "det"}});
    cfg.web_root = web;
    ServingServer srv(cfg, fake_model_builder());
    std::string err;
    REQUIRE(srv.start(&err));
    preload(srv, "det");
    httplib::Client cli("127.0.0.1", srv.port());

    auto idx = cli.Get("/");
    REQUIRE((idx && idx->status == 200));
    REQUIRE(idx->body.find("<h1>ok</h1>") != std::string::npos);

    auto js = cli.Get("/app.js");
    REQUIRE((js && js->status == 200));
    auto svg = cli.Get("/assets/logo.svg");
    REQUIRE((svg && svg->status == 200));

    // 缺失文件 → 404（不落到 API）
    auto miss = cli.Get("/nope.txt");
    REQUIRE((miss && miss->status == 404));

    // 路径穿越 → 404
    auto trav = cli.Get("/../CMakeLists.txt");
    REQUIRE((trav && trav->status == 404));

    // API 路由不被静态托管遮蔽
    auto api = cli.Get("/v1/models");
    REQUIRE((api && api->status == 200));

    auto health = cli.Get("/health");
    REQUIRE((health && health->status == 200));

    fs::remove_all(web);
    fs::remove_all(repo);
}

TEST_CASE("serving lazy: manifest parses entries", "[serving]") {
    auto dir = make_temp_repo();
    fs::create_directories(fs::path(dir) / "assets");
    { std::ofstream f(fs::path(dir) / "assets" / "coco.txt"); f << "person\ncar\n"; }
    nlohmann::json j;
    j["base"] = dir;
    j["models"] = nlohmann::json::array();
    j["models"].push_back({{"id", "det"}, {"display", "Det"}, {"type", "det"},
                           {"files", {{"model", "yolo11n/model.onnx"}}},
                           {"labels", {"person", "car"}}, {"input_size", {640, 640}},
                           {"desc", "detect"}});
    j["models"].push_back({{"id", "cls"}, {"type", "cls"},
                           {"files", {{"model", "yolo11n-cls/model.onnx"}}},
                           {"labels", "assets/coco.txt"}});
    j["models"].push_back({{"id", "ocr"}, {"type", "ocr"},
                           {"files", {{"rec", "ppocrv4/rec.onnx"},
                                      {"cls", "ppocrv4/cls.onnx"},
                                      {"dict", "ppocrv4/ppocr_keys.txt"}}}});
    auto path = (fs::path(dir) / "manifest.json").string();
    { std::ofstream f(path); f << j.dump(); }

    std::vector<ManifestModel> out;
    std::string err;
    REQUIRE(load_manifest(path, dir, &out, &err));
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].id == "det");
    REQUIRE(out[0].type == "det");
    REQUIRE(out[0].display == "Det");
    REQUIRE(out[0].labels.size() == 2);
    REQUIRE(out[0].input_size == std::vector<int>{640, 640});
    REQUIRE(out[0].model_f == (fs::path(dir) / "yolo11n/model.onnx").string());
    REQUIRE(out[1].labels.size() == 2);                        // 从 assets/coco.txt 读入
    REQUIRE(out[1].labels[0] == "person");
    REQUIRE(out[1].input_size == std::vector<int>{640, 640});  // 缺省 640x640
    REQUIRE(out[1].model_f == (fs::path(dir) / "yolo11n-cls/model.onnx").string());
    REQUIRE(out[2].rec_f == (fs::path(dir) / "ppocrv4/rec.onnx").string());
    REQUIRE(out[2].cls_f == (fs::path(dir) / "ppocrv4/cls.onnx").string());
    REQUIRE(out[2].dict_f == (fs::path(dir) / "ppocrv4/ppocr_keys.txt").string());

    fs::remove_all(dir);
}

TEST_CASE("serving lazy: startup catalog not instantiated", "[serving]") {
    auto dir = make_temp_repo();
    nlohmann::json j;
    j["base"] = dir;
    j["models"] = nlohmann::json::array();
    j["models"].push_back({{"id", "det"}, {"type", "det"}});
    j["models"].push_back({{"id", "cls"}, {"type", "cls"}});
    auto path = (fs::path(dir) / "manifest.json").string();
    { std::ofstream f(path); f << j.dump(); }

    int builds = 0;
    HandleBuilder b = [&](const ManifestModel&, const std::string&) {
        ++builds;
        ModelHandle h;
        h.infer = [](const nlohmann::json&, nlohmann::json*, std::string*) { return true; };
        return h;
    };
    ServingConfig cfg;
    cfg.model_repo = path;
    cfg.web_root = "";
    ModelRepo m(cfg, std::move(b));
    m.scan();
    REQUIRE(builds == 0);  // 启动不实例化
    auto list = m.list();
    REQUIRE(list.size() >= 1);
    REQUIRE(list[0].status == ModelStatus::Unloaded);
    fs::remove_all(dir);
}

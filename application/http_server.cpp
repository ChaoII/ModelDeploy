#include "http_server.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::json;

HttpServer::HttpServer(PipelineManager& mgr, const std::string& host, int port)
    : mgr_(mgr), host_(host), port_(port) {
}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start() {
    if (running_) return true;
    register_routes();

    running_ = true;
    server_thread_ = std::thread([this]() {
        std::cout << "[HttpServer] Starting on " << host_ << ":" << port_ << " ..." << std::endl;
        if (!server_.listen(host_.c_str(), port_)) {
            std::cerr << "[HttpServer] FAILED to bind " << host_ << ":" << port_
                      << " - port may be in use by another process" << std::endl;
            running_ = false;
        } else {
            std::cout << "[HttpServer] Listening on " << host_ << ":" << port_ << std::endl;
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!running_) {
        std::cerr << "[HttpServer] Could not start. Port " << port_ << " may be occupied." << std::endl;
    }
    return running_.load();
}

void HttpServer::stop() {
    running_ = false;
    server_.stop();
    if (server_thread_.joinable())
        server_thread_.join();
    std::cout << "[HttpServer] Stopped" << std::endl;
}

// ── JSON helpers ──────────────────────────────

std::string HttpServer::err_json(const std::string& msg) {
    return json{{"ok", false}, {"msg", msg}}.dump();
}

std::string HttpServer::ok_json(const json& data) {
    json r{{"ok", true}};
    if (data.is_object()) {
        for (auto& [k, v] : data.items()) r[k] = v;
    }
    return r.dump();
}

json HttpServer::task_status_to_json(const TaskStatus& ts) {
    json j;
    j["id"] = ts.id;
    j["name"] = ts.name;
    j["running"] = ts.running;
    j["initialized"] = ts.initialized;
    j["init_error"] = ts.init_error;
    j["input_url"] = ts.input_url;
    j["output_url"] = ts.output_url;
    j["preview_url"] = ts.preview_url;
    j["enable_preview"] = ts.enable_preview;
    json models = json::array();
    for (size_t i = 0; i < ts.model_names.size(); ++i) {
        json m{{"name", ts.model_names[i]}, {"type", i < ts.model_types.size() ? ts.model_types[i] : ""}};
        models.push_back(m);
    }
    j["models"] = models;
    try {
        j["stats"] = json::parse(ts.stats_json);
    } catch (...) {
        j["stats"] = json::object();
    }
    return j;
}

json HttpServer::model_config_to_json(const ModelConfig& m) {
    json j;
    j["name"] = m.name;
    j["type"] = m.type;
    j["path"] = m.path;
    j["backend"] = m.backend;
    j["device"] = m.device;
    j["confidence_threshold"] = m.confidence_threshold;
    j["input_size"] = {m.input_size[0], m.input_size[1]};
    j["roi"] = {m.roi[0], m.roi[1], m.roi[2], m.roi[3]};
    j["interval"] = m.interval;
    if (!m.rec_path.empty()) j["rec_path"] = m.rec_path;
    if (!m.labels.empty()) j["labels"] = m.labels;
    return j;
}

static std::string get_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buf, MAX_PATH) > 0) {
        std::string path(buf);
        auto pos = path.find_last_of("\\/");
        if (pos != std::string::npos) return path.substr(0, pos);
    }
#endif
    return "";
}

std::string HttpServer::load_web_ui() const {
    std::vector<std::string> paths;
    auto exe_dir = get_exe_dir();
    if (!exe_dir.empty()) {
        paths.push_back(exe_dir + "\\web_ui.html");
        paths.push_back(exe_dir + "\\..\\application\\web_ui.html");
        paths.push_back(exe_dir + "\\..\\..\\application\\web_ui.html");
    }
    paths.push_back("E:\\CLionProjects\\ModelDeploy\\application\\web_ui.html");
    paths.push_back("web_ui.html");
    paths.push_back("../application/web_ui.html");
    paths.push_back("application/web_ui.html");
    for (const auto& p : paths) {
        std::ifstream f(p, std::ios::binary);
        if (f.is_open()) {
            std::stringstream buf;
            buf << f.rdbuf();
            auto s = buf.str();
            if (!s.empty()) return s;
        }
    }
    std::cerr << "[WebUI] Could not open web_ui.html" << std::endl;
    return "<h1>Web UI file not found</h1>";
}

// ── Route helpers ─────────────────────────────

static std::string get_id(const httplib::Request& req) {
    auto it = req.path_params.find("id");
    return it != req.path_params.end() ? it->second : "";
}

static ModelConfig parse_model_config(const std::string& body) {
    ModelConfig mcfg;
    try {
        auto j = json::parse(body);
        if (!j.is_object()) return mcfg;
        auto get_s = [&](const std::string& k, const std::string& d = "") -> std::string {
            return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : d;
        };
        mcfg.name = get_s("name");
        mcfg.path = get_s("path");
        mcfg.type = get_s("type", "detection");
        mcfg.backend = get_s("backend", "ort");
        mcfg.device = get_s("device", "gpu");
        if (j.contains("confidence_threshold") && j["confidence_threshold"].is_number())
            mcfg.confidence_threshold = j["confidence_threshold"];
        if (j.contains("input_size") && j["input_size"].is_array() && j["input_size"].size() >= 2)
            mcfg.input_size = {j["input_size"][0], j["input_size"][1]};
        if (j.contains("roi") && j["roi"].is_array() && j["roi"].size() >= 4)
            mcfg.roi = {j["roi"][0], j["roi"][1], j["roi"][2], j["roi"][3]};
        if (j.contains("interval") && j["interval"].is_number_integer())
            mcfg.interval = j["interval"];
        if (j.contains("rec_path")) mcfg.rec_path = get_s("rec_path");
        if (j.contains("labels") && j["labels"].is_array())
            mcfg.labels = j["labels"].get<std::vector<std::string>>();
    } catch (...) {}
    return mcfg;
}

// ── Routes ────────────────────────────────────

void HttpServer::register_routes() {

    // ── Web UI ──────────────────────────────────────
    server_.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(load_web_ui(), "text/html; charset=utf-8");
    });
    server_.Get("/index.html", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(load_web_ui(), "text/html; charset=utf-8");
    });
    server_.Get("/ui", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(load_web_ui(), "text/html; charset=utf-8");
    });

    // ── Health / Metrics ─────────────────────────────
    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(ok_json({{"status", "ok"}, {"time", std::time(nullptr)}}), "application/json");
    });

    server_.Get("/api/v1/metrics", [this](const httplib::Request&, httplib::Response& res) {
        auto tasks = mgr_.list_tasks();
        int running = 0, initialized = 0, error = 0;
        for (const auto& t : tasks) {
            if (t.running) ++running;
            if (t.initialized) ++initialized;
            if (!t.init_error.empty()) ++error;
        }
        json j;
        j["tasks_total"] = tasks.size();
        j["tasks_running"] = running;
        j["tasks_initialized"] = initialized;
        j["tasks_error"] = error;
        j["models_total"] = mgr_.list_models().size();
        j["time"] = std::time(nullptr);
        res.set_content(ok_json({{"metrics", j}}), "application/json");
    });

    server_.Post("/api/v1/save", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(ok_json({{"saved", true}}), "application/json");
    });

    // ── POST /api/v1/tasks — create ─────────────────
    server_.Post("/api/v1/tasks", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto cfg = task_config_from_json(json::parse(req.body));
            std::cout << "[API] POST /api/v1/tasks  id=" << cfg.id
                      << " input=" << cfg.input_url
                      << " output=" << cfg.output_url
                      << " models=" << cfg.models.size() << std::endl;

            std::string err;
            if (!mgr_.create_task(cfg, &err)) {
                std::cerr << "[API] Create task failed: " << cfg.id << " - " << err << std::endl;
                res.set_content(err_json(err), "application/json");
                return;
            }
            std::cout << "[API] Task created: " << cfg.id << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            std::cerr << "[API] Create task parse error: " << e.what() << std::endl;
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── GET /api/v1/tasks — list ────────────────────
    server_.Get("/api/v1/tasks", [this](const httplib::Request&, httplib::Response& res) {
        auto tasks = mgr_.list_tasks();
        json arr = json::array();
        for (const auto& t : tasks) arr.push_back(task_status_to_json(t));
        res.set_content(ok_json({{"tasks", arr}}), "application/json");
    });

    // ── GET /api/v1/tasks/:id — config ──────────────
    server_.Get("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        TaskConfig cfg;
        if (!mgr_.get_task_config(id, &cfg)) {
            res.set_content(err_json("task not found: " + id), "application/json");
            return;
        }
        res.set_content(ok_json({{"task", task_config_to_json(cfg)}}), "application/json");
    });

    // ── GET /api/v1/tasks/:id/stats ─────────────────
    server_.Get("/api/v1/tasks/:id/stats", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        std::string stats_json;
        if (!mgr_.get_task_stats(id, &stats_json)) {
            res.set_content(err_json("task not found: " + id), "application/json");
            return;
        }
        try {
            res.set_content(ok_json({{"stats", json::parse(stats_json)}}), "application/json");
        } catch (...) {
            res.set_content(ok_json({{"stats", json::object()}}), "application/json");
        }
    });

    // ── GET /api/v1/tasks/:id/snapshot.jpg — 最新一帧 JPEG ──
    server_.Get("/api/v1/tasks/:id/snapshot.jpg", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        std::vector<uint8_t> jpg;
        int quality = 80;
        if (req.has_param("q")) {
            try { quality = std::stoi(req.get_param_value("q")); } catch (...) {}
        }
        if (!mgr_.get_task_jpeg(id, &jpg, quality) || jpg.empty()) {
            res.status = 404;
            res.set_content("snapshot unavailable", "text/plain");
            return;
        }
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("X-Task-Id", id);
        res.set_content(reinterpret_cast<const char*>(jpg.data()), jpg.size(), "image/jpeg");
    });

    // ── DELETE /api/v1/tasks/:id — remove ───────────
    server_.Delete("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        std::cout << "[API] DELETE /api/v1/tasks/" << id << std::endl;
        if (!mgr_.remove_task(id)) {
            std::cerr << "[API] Delete failed: task not found: " << id << std::endl;
            res.set_content(err_json("task not found"), "application/json");
            return;
        }
        std::cout << "[API] Task deleted: " << id << std::endl;
        res.set_content(ok_json(), "application/json");
    });

    // ── POST /api/v1/tasks/:id/start ────────────────
    server_.Post("/api/v1/tasks/:id/start", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        std::cout << "[API] POST /api/v1/tasks/" << id << "/start" << std::endl;
        if (!mgr_.start_task(id)) {
            std::cerr << "[API] Start failed for " << id << std::endl;
            res.set_content(err_json("task not found"), "application/json");
            return;
        }
        std::cout << "[API] Task " << id << " starting (async initialization)" << std::endl;
        res.set_content(ok_json(), "application/json");
    });

    // ── POST /api/v1/tasks/:id/stop ─────────────────
    server_.Post("/api/v1/tasks/:id/stop", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        std::cout << "[API] POST /api/v1/tasks/" << id << "/stop" << std::endl;
        if (!mgr_.stop_task(id)) {
            std::cerr << "[API] Stop failed: task not found: " << id << std::endl;
            res.set_content(err_json("task not found"), "application/json");
            return;
        }
        std::cout << "[API] Task " << id << " stopped" << std::endl;
        res.set_content(ok_json(), "application/json");
    });

    // ── PUT /api/v1/tasks/:id — update config ───────
    server_.Put("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        try {
            auto j = json::parse(req.body);
            auto cfg = task_config_from_json(j);
            cfg.id = id; // 保持原有 id
            // 若请求体包含 models 则替换模型列表；否则保留现有模型
            TaskConfig old_cfg;
            if (mgr_.get_task_config(id, &old_cfg)) {
                if (!j.contains("models")) {
                    cfg.models = old_cfg.models;
                }
            }
            std::cout << "[API] PUT /api/v1/tasks/" << id
                      << " enable_preview=" << cfg.enable_preview << std::endl;
            if (j.contains("models")) {
                std::cout << "  models: " << cfg.models.size() << std::endl;
            }
            std::string err;
            if (!mgr_.update_task(id, cfg, &err)) {
                res.set_content(err_json(err), "application/json");
                return;
            }
            std::cout << "[API] Task config updated: " << id << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── PATCH /api/v1/tasks/:id/models — add model ─
    server_.Patch("/api/v1/tasks/:id/models", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        try {
            auto mcfg = parse_model_config(req.body);
            if (mcfg.name.empty()) {
                res.set_content(err_json("model name required"), "application/json");
                return;
            }
            std::cout << "[API] PATCH /api/v1/tasks/" << id << "/models  add=" << mcfg.name << std::endl;
            if (!mgr_.add_model(id, mcfg)) {
                res.set_content(err_json("add model failed"), "application/json");
                return;
            }
            std::cout << "[API] Model " << mcfg.name << " added to " << id << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── DELETE /api/v1/tasks/:id/models/:name ───────
    server_.Delete("/api/v1/tasks/:id/models/:name",
        [this](const httplib::Request& req, httplib::Response& res) {
        auto id = req.path_params.find("id");
        auto name = req.path_params.find("name");
        if (id == req.path_params.end() || name == req.path_params.end()) {
            res.set_content(err_json("missing params"), "application/json");
            return;
        }
        std::cout << "[API] DELETE /api/v1/tasks/" << id->second << "/models/" << name->second << std::endl;
        if (!mgr_.remove_model(id->second, name->second)) {
            res.set_content(err_json("remove model failed"), "application/json");
            return;
        }
        std::cout << "[API] Model " << name->second << " removed from " << id->second << std::endl;
        res.set_content(ok_json(), "application/json");
    });

    // ── PATCH /api/v1/tasks/:id/models/:name ────────
    server_.Patch("/api/v1/tasks/:id/models/:name",
        [this](const httplib::Request& req, httplib::Response& res) {
        auto id = req.path_params.find("id");
        auto name = req.path_params.find("name");
        if (id == req.path_params.end() || name == req.path_params.end()) {
            res.set_content(err_json("missing params"), "application/json");
            return;
        }
        try {
            auto mcfg = parse_model_config(req.body);
            mcfg.name = name->second;
            std::cout << "[API] PATCH /api/v1/tasks/" << id->second << "/models/" << name->second << std::endl;
            if (!mgr_.update_model(id->second, name->second, mcfg)) {
                res.set_content(err_json("update model failed"), "application/json");
                return;
            }
            std::cout << "[API] Model " << name->second << " updated on " << id->second << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── GET /api/v1/models — list model library ────
    server_.Get("/api/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = mgr_.list_models();
        json arr = json::array();
        for (const auto& m : models) arr.push_back(model_config_to_json(m));
        res.set_content(ok_json({{"models", arr}}), "application/json");
    });

    // ── POST /api/v1/models — add model to library ──
    server_.Post("/api/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto mcfg = parse_model_config(req.body);
            if (mcfg.name.empty()) {
                res.set_content(err_json("model name required"), "application/json");
                return;
            }
            std::cout << "[API] POST /api/v1/models  name=" << mcfg.name << std::endl;
            if (!mgr_.add_model_to_library(mcfg)) {
                res.set_content(err_json("add failed (name may exist)"), "application/json");
                return;
            }
            std::cout << "[API] Model added to library: " << mcfg.name << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── PATCH /api/v1/models/:name — update in library ──
    server_.Patch("/api/v1/models/:name", [this](const httplib::Request& req, httplib::Response& res) {
        auto it = req.path_params.find("name");
        std::string name = (it != req.path_params.end()) ? it->second : "";
        try {
            auto mcfg = parse_model_config(req.body);
            std::cout << "[API] PATCH /api/v1/models/" << name << std::endl;
            if (!mgr_.update_model_in_library(name, mcfg)) {
                res.set_content(err_json("model not found"), "application/json");
                return;
            }
            std::cout << "[API] Model updated in library: " << name << std::endl;
            res.set_content(ok_json(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(err_json(e.what()), "application/json");
        }
    });

    // ── DELETE /api/v1/models/:name ─ remove from library ──
    server_.Delete("/api/v1/models/:name", [this](const httplib::Request& req, httplib::Response& res) {
        auto it = req.path_params.find("name");
        std::string name = (it != req.path_params.end()) ? it->second : "";
        std::cout << "[API] DELETE /api/v1/models/" << name << std::endl;
        if (!mgr_.remove_model_from_library(name)) {
            res.set_content(err_json("model not found"), "application/json");
            return;
        }
        std::cout << "[API] Model removed from library: " << name << std::endl;
        res.set_content(ok_json(), "application/json");
    });
}

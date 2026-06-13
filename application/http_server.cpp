#include "http_server.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

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
    return "{\"ok\":false,\"msg\":\"" + msg + "\"}";
}

std::string HttpServer::ok_json(const std::string& data) {
    if (data.empty() || data[0] != '{')
        return "{\"ok\":true}";
    return "{\"ok\":true," + data.substr(1);
}

std::string HttpServer::task_list_json(const std::vector<TaskStatus>& tasks) {
    std::string j = "{\"ok\":true,\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i) j += ",";
        j += "{\"id\":\"" + tasks[i].id + "\",\"name\":\"" + tasks[i].name
          + "\",\"running\":" + (tasks[i].running ? "true" : "false")
          + ",\"input_url\":\"" + tasks[i].input_url
          + "\",\"output_url\":\"" + tasks[i].output_url + "\"";
        if (!tasks[i].model_names.empty()) {
            j += ",\"models\":[";
            for (size_t k = 0; k < tasks[i].model_names.size(); ++k) {
                if (k) j += ",";
                j += "{\"name\":\"" + tasks[i].model_names[k]
                  + "\",\"type\":\"" + tasks[i].model_types[k] + "\"}";
            }
            j += "]";
        }
        j += "}";
    }
    j += "]}";
    return j;
}

std::string HttpServer::task_config_json(const TaskConfig& cfg) {
    return "{\"ok\":true,\"task\":" + task_config_to_json(cfg).dump() + "}";
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
        if (j.contains("confidence_threshold")) mcfg.confidence_threshold = j["confidence_threshold"];
        if (j.contains("input_size") && j["input_size"].is_array() && j["input_size"].size() >= 2)
            mcfg.input_size = {j["input_size"][0], j["input_size"][1]};
        if (j.contains("roi") && j["roi"].is_array() && j["roi"].size() >= 4)
            mcfg.roi = {j["roi"][0], j["roi"][1], j["roi"][2], j["roi"][3]};
        if (j.contains("interval")) mcfg.interval = j["interval"];
    } catch (...) {}
    return mcfg;
}

// Check HTTP method match
static bool is_method(const httplib::Request& req, const std::string& method) {
    return req.method == method;
}

// ── Routes ────────────────────────────────────

void HttpServer::register_routes() {

    // ── Web UI (served from file) ───────────────────
    server_.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        // Try multiple possible locations for the HTML file
        std::vector<std::string> paths = {
            "E:\\CLionProjects\\ModelDeploy\\application\\web_ui.html",
            "web_ui.html",
            "../application/web_ui.html"
        };
        for (const auto& p : paths) {
            std::ifstream f(p);
            if (f.is_open()) {
                try {
                    std::stringstream buf;
                    buf << f.rdbuf();
                    std::string html = buf.str();
                    if (html.empty()) {
                        std::cerr << "[WebUI] File " << p << " is empty" << std::endl;
                        continue;
                    }
                    res.set_content(html, "text/html; charset=utf-8");
                    return;
                } catch (const std::exception& e) {
                    std::cerr << "[WebUI] Error reading " << p << ": " << e.what() << std::endl;
                    continue;
                }
            }
        }
        std::cerr << "[WebUI] Could not open web_ui.html from any path" << std::endl;
        res.set_content("<h1>Web UI file not found</h1><p>Expected at: E:\\CLionProjects\\ModelDeploy\\application\\web_ui.html</p>", "text/html");
    });

    // ── Health ──────────────────────────────────────
    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"ok\":true}", "application/json");
    });

    // ── POST /api/v1/tasks — create ─────────────────
    server_.Post("/api/v1/tasks", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto cfg = task_config_from_json(json::parse(req.body));
            std::cout << "[API] POST /api/v1/tasks  id=" << cfg.id
                      << " input=" << cfg.input_url
                      << " output=" << cfg.output_url
                      << " models=" << cfg.models.size() << std::endl;
            if (!mgr_.create_task(cfg)) {
                std::cerr << "[API] Create task failed: " << cfg.id << " (may already exist)" << std::endl;
                res.set_content(err_json("create failed: task may already exist"), "application/json");
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
        res.set_content(task_list_json(tasks), "application/json");
    });

    // ── GET /api/v1/tasks/:id — config ──────────────
    server_.Get("/api/v1/tasks/:id", [this](const httplib::Request& req, httplib::Response& res) {
        std::string id = get_id(req);
        TaskConfig cfg;
        if (!mgr_.get_task_config(id, &cfg)) {
            res.set_content(err_json("task not found: " + id), "application/json");
            return;
        }
        res.set_content(ok_json("\"task\":" + task_config_to_json(cfg).dump()), "application/json");
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
            std::cerr << "[API] Start failed for " << id
                      << " (pipeline init will continue in background)" << std::endl;
            // start() now returns true even if init fails (async)
            // So if it returns false, the task doesn't exist
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
}

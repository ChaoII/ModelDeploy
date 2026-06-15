#include "pipeline_manager.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

PipelineManager::~PipelineManager() {
    stop_all();
}

bool PipelineManager::create_task(const TaskConfig& cfg, std::string* err) {
    std::string local_err;
    if (!cfg.validate(&local_err)) {
        if (err) *err = local_err;
        std::cerr << "[Manager] Invalid task config: " << local_err << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (cfg.id.empty()) {
        if (err) *err = "task id is empty";
        return false;
    }
    if (pipelines_.find(cfg.id) != pipelines_.end()) {
        std::cerr << "[Manager] Task already exists: " << cfg.id << std::endl;
        if (err) *err = "task already exists";
        return false;
    }
    // 每路独立加载模型实例（依赖 TRT engine cache 共享编译产物）
    // 解码器通过 StreamHub 共享：相同 url+config 的多路任务复用同一解码器
    pipelines_[cfg.id] = std::make_unique<Pipeline>(cfg, &stream_hub_);
    dirty_ = true;
    std::cout << "[Manager] Task created: " << cfg.id << std::endl;
    return true;
}

bool PipelineManager::remove_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    // unique_ptr destructor calls Pipeline::stop()
    pipelines_.erase(it);
    dirty_ = true;
    std::cout << "[Manager] Task removed: " << task_id << std::endl;
    return true;
}

bool PipelineManager::start_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    return it->second->start();
}

bool PipelineManager::stop_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    it->second->stop();
    return true;
}

bool PipelineManager::get_task_config(const std::string& task_id, TaskConfig* cfg) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end() || !cfg) return false;
    *cfg = it->second->config();
    return true;
}

bool PipelineManager::get_task_stats(const std::string& task_id, std::string* stats_json) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end() || !stats_json) return false;
    *stats_json = it->second->stats().to_json();
    return true;
}

bool PipelineManager::get_task_jpeg(const std::string& task_id, std::vector<uint8_t>* jpeg, int quality) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end() || !jpeg) return false;
    return it->second->latest_jpeg(jpeg, quality);
}

std::vector<TaskStatus> PipelineManager::list_tasks() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<TaskStatus> result;
    result.reserve(pipelines_.size());
    for (const auto& [id, pipe] : pipelines_) {
        TaskStatus ts;
        ts.id = id;
        ts.name = pipe->config().name;
        ts.running = pipe->is_running();
        ts.initialized = pipe->is_initialized();
        ts.init_error = pipe->init_error();
        ts.input_url = pipe->config().input_url;
        ts.output_url = pipe->config().output_url;
        ts.preview_url = pipe->config().preview_url;
        for (const auto& m : pipe->config().models) {
            ts.model_names.push_back(m.name);
            ts.model_types.push_back(m.type);
        }
        ts.stats_json = pipe->stats().to_json();
        result.push_back(ts);
    }
    return result;
}

size_t PipelineManager::task_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pipelines_.size();
}

bool PipelineManager::add_model(const std::string& task_id, const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    bool ok = it->second->add_model(mcfg);
    if (ok) dirty_ = true;
    return ok;
}

bool PipelineManager::remove_model(const std::string& task_id, const std::string& model_name) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    bool ok = it->second->remove_model(model_name);
    if (ok) dirty_ = true;
    return ok;
}

bool PipelineManager::update_model(const std::string& task_id, const std::string& model_name,
                                    const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    bool ok = it->second->update_model(model_name, mcfg);
    if (ok) dirty_ = true;
    return ok;
}

void PipelineManager::stop_all() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [id, pipe] : pipelines_) {
        pipe->stop();
    }
    pipelines_.clear();
    dirty_ = true;
}

// ── 模型库管理 ──

bool PipelineManager::add_model_to_library(const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (mcfg.name.empty()) return false;
    // 检查名字是否已存在
    for (const auto& m : model_library_)
        if (m.name == mcfg.name) return false;
    model_library_.push_back(mcfg);
    dirty_ = true;
    std::cout << "[ModelLib] Added: " << mcfg.name << " [" << mcfg.type << "] " << mcfg.path << std::endl;
    return true;
}

bool PipelineManager::remove_model_from_library(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = model_library_.begin(); it != model_library_.end(); ++it) {
        if (it->name == name) {
            model_library_.erase(it);
            dirty_ = true;
            std::cout << "[ModelLib] Removed: " << name << std::endl;
            return true;
        }
    }
    return false;
}

std::vector<ModelConfig> PipelineManager::list_models() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return model_library_;
}

bool PipelineManager::get_model(const std::string& name, ModelConfig* mcfg) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& m : model_library_) {
        if (m.name == name) {
            if (mcfg) *mcfg = m;
            return true;
        }
    }
    return false;
}

bool PipelineManager::update_model_in_library(const std::string& name, const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& m : model_library_) {
        if (m.name == name) {
            m = mcfg;
            m.name = name; // 保持原名称
            dirty_ = true;
            std::cout << "[ModelLib] Updated: " << name << std::endl;
            return true;
        }
    }
    return false;
}

// ── 持久化 ──

bool PipelineManager::load_from_directory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mtx_);
    try {
        fs::path p(dir);
        fs::create_directories(p);

        // 1) 加载模型库
        fs::path models_path = p / "models.json";
        if (fs::exists(models_path)) {
            std::ifstream f(models_path);
            if (f.is_open()) {
                json j;
                f >> j;
                if (j.contains("models") && j["models"].is_array()) {
                    for (const auto& mo : j["models"]) {
                        ModelConfig m;
                        if (mo.contains("name")) m.name = mo["name"];
                        if (mo.contains("type")) m.type = mo["type"];
                        if (mo.contains("path")) m.path = mo["path"];
                        if (mo.contains("rec_path")) m.rec_path = mo["rec_path"];
                        if (mo.contains("backend")) m.backend = mo["backend"];
                        if (mo.contains("device")) m.device = mo["device"];
                        if (mo.contains("confidence_threshold")) m.confidence_threshold = mo["confidence_threshold"];
                        if (mo.contains("input_size") && mo["input_size"].is_array() && mo["input_size"].size() >= 2)
                            m.input_size = {mo["input_size"][0], mo["input_size"][1]};
                        if (mo.contains("roi") && mo["roi"].is_array() && mo["roi"].size() >= 4)
                            m.roi = {mo["roi"][0], mo["roi"][1], mo["roi"][2], mo["roi"][3]};
                        if (mo.contains("interval")) m.interval = mo["interval"];
                        if (mo.contains("labels") && mo["labels"].is_array())
                            m.labels = mo["labels"].get<std::vector<std::string>>();
                        model_library_.push_back(m);
                    }
                    std::cout << "[Persistence] Loaded " << model_library_.size() << " models" << std::endl;
                }
            }
        }

        // 2) 加载任务（只加载配置，不自动启动）
        fs::path tasks_path = p / "tasks.json";
        if (fs::exists(tasks_path)) {
            std::ifstream f(tasks_path);
            if (f.is_open()) {
                json j;
                f >> j;
                if (j.contains("tasks") && j["tasks"].is_array()) {
                    int count = 0;
                    for (const auto& t : j["tasks"]) {
                        auto cfg = task_config_from_json(t);
                        if (cfg.validate()) {
                            pipelines_[cfg.id] = std::make_unique<Pipeline>(cfg);
                            ++count;
                        }
                    }
                    std::cout << "[Persistence] Loaded " << count << " tasks" << std::endl;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Persistence] Load failed: " << e.what() << std::endl;
        return false;
    }
}

bool PipelineManager::save_to_directory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mtx_);
    try {
        fs::path p(dir);
        fs::create_directories(p);

        // 1) 保存模型库
        {
            json j;
            j["models"] = json::array();
            for (const auto& m : model_library_) {
                json mo;
                mo["name"] = m.name;
                mo["type"] = m.type;
                mo["path"] = m.path;
                mo["backend"] = m.backend;
                mo["device"] = m.device;
                mo["confidence_threshold"] = m.confidence_threshold;
                mo["input_size"] = {m.input_size[0], m.input_size[1]};
                mo["roi"] = {m.roi[0], m.roi[1], m.roi[2], m.roi[3]};
                mo["interval"] = m.interval;
                if (!m.rec_path.empty()) mo["rec_path"] = m.rec_path;
                if (!m.labels.empty()) mo["labels"] = m.labels;
                j["models"].push_back(mo);
            }
            std::ofstream f(p / "models.json");
            if (f.is_open()) f << j.dump(2);
        }

        // 2) 保存任务配置
        {
            json j;
            j["tasks"] = json::array();
            for (const auto& [id, pipe] : pipelines_) {
                j["tasks"].push_back(task_config_to_json(pipe->config()));
            }
        std::ofstream f(p / "tasks.json");
        if (f.is_open()) f << j.dump(2);
        }
        dirty_ = false;
        std::cout << "[Persistence] Saved to " << dir << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Persistence] Save failed: " << e.what() << std::endl;
        return false;
    }
}

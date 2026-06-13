#include "pipeline_manager.hpp"
#include <iostream>

PipelineManager::~PipelineManager() {
    stop_all();
}

bool PipelineManager::create_task(const TaskConfig& cfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (cfg.id.empty()) return false;
    if (pipelines_.find(cfg.id) != pipelines_.end()) {
        std::cerr << "[Manager] Task already exists: " << cfg.id << std::endl;
        return false;
    }
    pipelines_[cfg.id] = std::make_unique<Pipeline>(cfg);
    return true;
}

bool PipelineManager::remove_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    // unique_ptr destructor calls Pipeline::stop()
    pipelines_.erase(it);
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

std::vector<TaskStatus> PipelineManager::list_tasks() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<TaskStatus> result;
    result.reserve(pipelines_.size());
    for (const auto& [id, pipe] : pipelines_) {
        TaskStatus ts;
        ts.id = id;
        ts.name = pipe->config().name;
        ts.running = pipe->is_running();
        ts.input_url = pipe->config().input_url;
        ts.output_url = pipe->config().output_url;
        for (const auto& m : pipe->config().models) {
            ts.model_names.push_back(m.name);
            ts.model_types.push_back(m.type);
        }
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
    return it->second->add_model(mcfg);
}

bool PipelineManager::remove_model(const std::string& task_id, const std::string& model_name) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    return it->second->remove_model(model_name);
}

bool PipelineManager::update_model(const std::string& task_id, const std::string& model_name,
                                    const ModelConfig& mcfg) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) return false;
    return it->second->update_model(model_name, mcfg);
}

void PipelineManager::stop_all() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [id, pipe] : pipelines_) {
        pipe->stop();
    }
    pipelines_.clear();
}

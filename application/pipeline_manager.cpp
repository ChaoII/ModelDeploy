#include "pipeline_manager.hpp"
#include "csrc/vision/face/face_det/scrfd.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace modeldeploy;
using namespace modeldeploy::vision;
using modeldeploy::vision::detection::UltralyticsDet;
using modeldeploy::vision::face::Scrfd;

// ── helper: 根据 ModelConfig 创建 RuntimeOption ──
static modeldeploy::RuntimeOption build_runtime_option(const ModelConfig& cfg) {
    modeldeploy::RuntimeOption opt;
    if (cfg.device == "gpu") opt.use_gpu(0);
    opt.set_cpu_thread_num(1);
    bool is_engine_file = (cfg.path.size() > 7 &&
        (cfg.path.substr(cfg.path.size() - 7) == ".engine" ||
         cfg.path.substr(cfg.path.size() - 7) == ".Engine"));
    if (is_engine_file || cfg.backend == "trt" || cfg.backend == "tensorrt") {
        opt.use_trt_backend();
        opt.enable_fp16 = true;
        std::string cache_dir = "data/trt_cache";
        try { std::filesystem::create_directories(cache_dir); } catch (...) {}
        std::string model_name = cfg.path.substr(cfg.path.find_last_of("/\\") + 1);
        opt.trt_option.cache_file_path = cache_dir + "/" + model_name + ".engine";
        opt.trt_option.enable_fp16 = true;
        opt.trt_option.max_workspace_size = 1ULL << 30;
        if (cfg.input_size.size() == 2) {
            std::string s = "1x3x" + std::to_string(cfg.input_size[0]) + "x" + std::to_string(cfg.input_size[1]);
            opt.set_trt_min_shape(s); opt.set_trt_opt_shape(s); opt.set_trt_max_shape(s);
        }
    } else if (cfg.backend == "mnn") {
        opt.use_mnn_backend();
    } else {
        opt.use_ort_backend();
        if (cfg.device == "gpu") {
            opt.enable_fp16 = true;
            opt.enable_trt = true;
            opt.ort_option.enable_trt = true;
            opt.ort_option.enable_fp16 = true;
            std::string cache_dir = "data/ort_trt_cache";
            try { std::filesystem::create_directories(cache_dir); } catch (...) {}
            opt.ort_option.trt_engine_cache_path = cache_dir;
            if (cfg.input_size.size() == 2) {
                int w = cfg.input_size[0], h = cfg.input_size[1];
                std::string shape = "images:1x3x" + std::to_string(h) + "x" + std::to_string(w);
                opt.ort_option.trt_min_shape = shape;
                opt.ort_option.trt_opt_shape = shape;
                opt.ort_option.trt_max_shape = shape;
            }
        }
    }
    return opt;
}

PipelineManager::PipelineManager() {
    // BatchScheduler 不自动启动，需要时手动调用 start_batch_scheduler()
}

PipelineManager::~PipelineManager() {
    stop_all();
    stop_batch_scheduler();
}

// ── 模型工厂（prototype 缓存 + clone 共享 Runtime） ──

std::unique_ptr<InferenceEngine> PipelineManager::create_engine(const ModelConfig& cfg) {
    // 分类模型不支持 clone，直接独立加载
    if (cfg.type == "classification") {
        auto eng = std::make_unique<InferenceEngine>();
        eng->load(cfg);
        return eng;
    }

    std::string key = InferenceEngine::make_cache_key(cfg);
    std::lock_guard<std::mutex> lock(proto_mtx_);

    // 检查是否已有 prototype
    auto it = model_prototypes_.find(key);
    if (it != model_prototypes_.end()) {
        auto& proto = it->second;
        auto eng = std::make_unique<InferenceEngine>();
        if (cfg.type == "detection" && proto.det && proto.det->is_initialized()) {
            eng->clone_detection_from(*proto.det, cfg);
            std::cout << "[Manager] Cloned detection: " << cfg.name << std::endl;
        } else if (cfg.type == "face_detection" && proto.face && proto.face->is_initialized()) {
            auto cloned = proto.face->clone(); // 共享 Runtime，独立 pre/post
            if (cloned && cloned->is_initialized()) {
                if (cfg.input_size.size() == 2)
                    cloned->get_preprocessor().set_size(cfg.input_size);
                if (cfg.device == "gpu")
                    cloned->get_preprocessor().use_cuda_preproc();
                eng->adopt_face_model(std::move(cloned), cfg);
                std::cout << "[Manager] Cloned face: " << cfg.name << std::endl;
            }
        }
        if (eng->is_loaded()) return eng;
    }

    // 首次加载
    auto opt = build_runtime_option(cfg);
    ModelPrototype proto;
    proto.key = key;

    if (cfg.type == "detection") {
        proto.det = std::make_unique<detection::UltralyticsDet>(cfg.path, opt);
        if (!proto.det->is_initialized()) {
            std::cerr << "[Manager] Failed to load detection: " << cfg.path << std::endl;
            auto eng = std::make_unique<InferenceEngine>(); eng->load(cfg);
            return eng;
        }
        if (cfg.input_size.size() == 2) proto.det->get_preprocessor().set_size(cfg.input_size);
        if (cfg.device == "gpu") proto.det->get_preprocessor().use_cuda_preproc();

        model_prototypes_[key] = std::move(proto);
        auto& lp = model_prototypes_[key].det;
        auto eng = std::make_unique<InferenceEngine>();
        eng->clone_detection_from(*lp, cfg);
        std::cout << "[Manager] Created detection prototype: " << cfg.name << std::endl;
        return eng;
    }

    if (cfg.type == "face_detection") {
        auto face_opt = build_runtime_option(cfg);
        using modeldeploy::vision::face::Scrfd;
        proto.face = std::unique_ptr<Scrfd>(new Scrfd(cfg.path, face_opt));
        if (!proto.face->is_initialized()) {
            std::cerr << "[Manager] Failed to load face: " << cfg.path << std::endl;
            auto eng = std::make_unique<InferenceEngine>(); eng->load(cfg);
            return eng;
        }
        if (cfg.input_size.size() == 2) proto.face->get_preprocessor().set_size(cfg.input_size);
        if (cfg.device == "gpu") proto.face->get_preprocessor().use_cuda_preproc();

        model_prototypes_[key] = std::move(proto);
        auto& lp = model_prototypes_[key].face;
        auto cloned = lp->clone();
        auto eng = std::make_unique<InferenceEngine>();
        eng->adopt_face_model(std::move(cloned), cfg);
        std::cout << "[Manager] Created face prototype: " << cfg.name << std::endl;
        return eng;
    }

    auto eng = std::make_unique<InferenceEngine>(); eng->load(cfg);
    return eng;
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

    // 模型工厂：通过 prototype cache + clone 共享 ORT Session（减少显存与 GPU 上下文切换）
    Pipeline::ModelFactory factory = [this](const ModelConfig& mcfg) {
        return this->create_engine(mcfg);
    };

    // 解码器通过 StreamHub 共享：相同 url+config 的多路任务复用同一解码器
    // 推理走独立 InferGroup 路径（不经过 BatchScheduler）
    pipelines_[cfg.id] = std::make_unique<Pipeline>(cfg, &stream_hub_, std::move(factory), nullptr);
    dirty_ = true;
    std::cout << "[Manager] Task created: " << cfg.id << " (no batch)" << std::endl;
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
        ts.enable_preview = pipe->config().enable_preview;
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

bool PipelineManager::update_task(const std::string& task_id, const TaskConfig& cfg, std::string* err) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    if (it == pipelines_.end()) {
        if (err) *err = "task not found";
        return false;
    }
    // 需要先停止才能修改配置
    if (it->second->is_running()) {
        if (err) *err = "task must be stopped before updating config";
        return false;
    }

    // Diff 模型列表：移除旧有的、添加新增的
    TaskConfig old_cfg = it->second->config();
    std::set<std::string> old_names, new_names;
    for (const auto& m : old_cfg.models) old_names.insert(m.name);
    for (const auto& m : cfg.models) new_names.insert(m.name);

    for (const auto& m : old_cfg.models) {
        if (!new_names.count(m.name)) {
            it->second->remove_model(m.name);
        }
    }
    for (const auto& m : cfg.models) {
        if (!old_names.count(m.name)) {
            it->second->add_model(m);
        }
    }

    bool ok = it->second->update_config(cfg);
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

bool PipelineManager::start_batch_scheduler() {
    bool ok = batch_scheduler_.start();
    if (ok) {
        // Register all models from the library
        for (const auto& m : model_library_) {
            batch_scheduler_.register_model(m);
        }
    }
    return ok;
}

void PipelineManager::stop_batch_scheduler() {
    batch_scheduler_.stop();
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
                            Pipeline::ModelFactory factory = [this](const ModelConfig& mcfg) {
                                return this->create_engine(mcfg);
                            };
                            pipelines_[cfg.id] = std::make_unique<Pipeline>(cfg, &stream_hub_, std::move(factory), nullptr);
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

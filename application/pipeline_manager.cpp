#include "pipeline_manager.hpp"
#include "runtime_factory.hpp"
#include "csrc/vision/face/face_det/scrfd.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>
#include <cstdlib>

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace modeldeploy;
using namespace modeldeploy::vision;
using modeldeploy::vision::detection::UltralyticsDet;
using modeldeploy::vision::face::Scrfd;

PipelineManager::PipelineManager() {}

PipelineManager::~PipelineManager() {
    stop_all();
}

// ── 模型工厂（prototype 缓存 + clone 共享 Runtime） ──

std::unique_ptr<InferenceEngine> PipelineManager::create_engine(const ModelConfig& cfg) {
    // classification / pose / obb / iseg / pedestrian_attribute / ocr / lpr 目前直接独立加载
    // （SDK 的模型支持 clone()，可后续接入 prototype 共享 Runtime；此处保守保持独立加载）
    if (cfg.type == "classification" || cfg.type == "pose" || cfg.type == "obb" ||
        cfg.type == "iseg" || cfg.type == "pedestrian_attribute" || cfg.type == "ocr" ||
        cfg.type == "lpr") {
        auto eng = std::make_unique<InferenceEngine>();
        eng->load(cfg);
        return eng;
    }

    std::string key = InferenceEngine::make_cache_key(cfg);
    std::lock_guard<std::mutex> lock(proto_mtx_);

    // 检查是否已有 prototype
    auto it = model_prototypes_.find(key);
    if (it != model_prototypes_.end()) {
        it->second.last_use = ++proto_tick_;   // 刷新 LRU
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
                eng->adopt_face_model(std::move(cloned), cfg);
                std::cout << "[Manager] Cloned face: " << cfg.name << std::endl;
            }
        }
        if (eng->is_loaded()) return eng;
    }

    // 首次加载
    ModelPrototype proto;
    proto.key = key;

    if (cfg.type == "detection") {
        // 共享批处理检测器：同配置多路共享一个模型实例 + AsyncModel 批推理（GPU 高利用率）
        auto det = get_or_create_detector(cfg);
        auto eng = std::make_unique<InferenceEngine>();
        eng->set_shared_detector(std::move(det), cfg);
        std::cout << "[Manager] Shared batched detector: " << cfg.name << std::endl;
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

        proto.last_use = ++proto_tick_;
        model_prototypes_[key] = std::move(proto);
        evict_prototypes_locked(key);   // 上限淘汰：防止不同配置反复热更新时无界驻留
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

std::shared_ptr<BatchedDetector> PipelineManager::get_or_create_detector(const ModelConfig& cfg) {
    // MD_DET_INDEPENDENT=1 或 TRT EP：每路独立检测器/session（不跨路共享）。TRT EP 动态 batch
    // 会按 batch size 惰性重建 engine，且多路并发共享单 session 争用更重；实测独立更优。
    // 其余情况共享池化（省显存 + 跨路批处理，适合 CUDA EP 等）。
    static const bool independent_env = std::getenv("MD_DET_INDEPENDENT") != nullptr;
    if (independent_env || cfg.use_trt_ep) {
        return std::make_shared<BatchedDetector>(cfg, /*max_batch=*/1);
    }
    const int w = cfg.input_size.size() >= 2 ? cfg.input_size[0] : 640;
    const int h = cfg.input_size.size() >= 2 ? cfg.input_size[1] : 640;
    const std::string key = cfg.path + "|" + cfg.backend + "|" + cfg.device + "|" +
                            std::to_string(cfg.confidence_threshold) + "|" +
                            std::to_string(w) + "x" + std::to_string(h);
    std::lock_guard<std::mutex> lk(det_pool_mtx_);
    // 先清理已失效条目：池用 weak_ptr 持有，使用者释放后条目即失效；
    // 这里顺带回收，避免 map 本身随热更新次数无界增长。
    for (auto it = det_pool_.begin(); it != det_pool_.end();) {
        it = it->second.expired() ? det_pool_.erase(it) : std::next(it);
    }
    auto it = det_pool_.find(key);
    if (it != det_pool_.end()) {
        if (auto live = it->second.lock()) return live;   // 仍有使用者 → 复用
        det_pool_.erase(it);                              // 竞态下刚失效 → 重新创建
    }
    auto det = std::make_shared<BatchedDetector>(cfg, /*max_batch=*/8);
    if (det->ok()) det_pool_[key] = det;
    return det;
}

size_t PipelineManager::detector_pool_size() const {
    std::lock_guard<std::mutex> lk(det_pool_mtx_);
    size_t n = 0;
    for (const auto& [k, w] : det_pool_)
        if (!w.expired()) ++n;
    return n;
}

size_t PipelineManager::prototype_count() const {
    std::lock_guard<std::mutex> lk(proto_mtx_);
    return model_prototypes_.size();
}

void PipelineManager::evict_prototypes_locked(const std::string& keep_key) {
    while (model_prototypes_.size() > kMaxPrototypes) {
        auto victim = model_prototypes_.end();
        for (auto it = model_prototypes_.begin(); it != model_prototypes_.end(); ++it) {
            if (it->first == keep_key) continue;
            if (victim == model_prototypes_.end() || it->second.last_use < victim->second.last_use)
                victim = it;
        }
        if (victim == model_prototypes_.end()) break;
        std::cout << "[Manager] Evicting model prototype (LRU, cap=" << kMaxPrototypes
                  << "): " << victim->first << std::endl;
        model_prototypes_.erase(victim);
    }
}

std::shared_ptr<PipelineManager::TaskEntry> PipelineManager::find_entry(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = pipelines_.find(task_id);
    return it == pipelines_.end() ? nullptr : it->second;
}

std::vector<std::pair<std::string, std::shared_ptr<PipelineManager::TaskEntry>>>
PipelineManager::snapshot_entries() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::pair<std::string, std::shared_ptr<TaskEntry>>> entries;
    entries.reserve(pipelines_.size());
    for (const auto& [id, e] : pipelines_) entries.emplace_back(id, e);
    return entries;
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

    auto entry = std::make_shared<TaskEntry>();
    entry->pipe = std::make_shared<Pipeline>(cfg, std::move(factory));
    pipelines_[cfg.id] = std::move(entry);
    dirty_ = true;
    std::cout << "[Manager] Task created: " << cfg.id << std::endl;
    return true;
}

bool PipelineManager::set_detection_sink(const std::string& task_id, Pipeline::DetectionSink sink) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;   // 加锁窗口内可能已被 remove
    entry->pipe->set_detection_sink(std::move(sink));
    return true;
}

bool PipelineManager::remove_task(const std::string& task_id) {
    std::shared_ptr<TaskEntry> entry;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = pipelines_.find(task_id);
        if (it == pipelines_.end()) return false;
        entry = it->second;
        pipelines_.erase(it);   // 先从映射摘除：新操作立即找不到本任务
        dirty_ = true;
    }
    // 等本任务在途慢操作结束，再停止（join 检测/编解码线程，可能秒级）——
    // 全程不持 mtx_：其它任务的启停/查询/调度不受影响。
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    entry->pipe->stop();
    std::cout << "[Manager] Task removed: " << task_id << std::endl;
    return true;
}

bool PipelineManager::start_task(const std::string& task_id) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    // 模型加载在 mtx_ 之外（旧实现持全局锁加载 → 控制面全线阻塞）
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;   // 加锁窗口内可能已被 remove
    return entry->pipe->start();
}

bool PipelineManager::stop_task(const std::string& task_id) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    // 线程 join 在 mtx_ 之外
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;
    entry->pipe->stop();
    return true;
}

bool PipelineManager::get_task_config(const std::string& task_id, TaskConfig* cfg) const {
    auto entry = find_entry(task_id);
    if (!entry || !cfg) return false;
    *cfg = entry->pipe->config();   // Pipeline::config() 内部按值快照（cfg_mtx_）
    return true;
}

bool PipelineManager::get_task_stats(const std::string& task_id, std::string* stats_json) const {
    auto entry = find_entry(task_id);
    if (!entry || !stats_json) return false;
    *stats_json = entry->pipe->stats().to_json();
    return true;
}

bool PipelineManager::get_task_jpeg(const std::string& task_id, std::vector<uint8_t>* jpeg, int quality) {
    if (!jpeg) return false;
    auto entry = find_entry(task_id);
    if (!entry) return false;
    // 仅做 shared_ptr 快照拷贝（微秒级），JPEG 编码在所有锁之外（5-15ms）
    std::shared_ptr<modeldeploy::vision::ImageData> snap;
    if (!entry->pipe->latest_bgr_snapshot(&snap)) return false;
    return Pipeline::encode_jpeg(snap, jpeg, quality);
}

std::vector<TaskStatus> PipelineManager::list_tasks() const {
    std::vector<TaskStatus> result;
    for (const auto& [id, entry] : snapshot_entries()) {
        const TaskConfig c = entry->pipe->config();   // 一次一致性快照
        TaskStatus ts;
        ts.id = id;
        ts.name = c.name;
        ts.running = entry->pipe->is_running();
        ts.initialized = entry->pipe->is_initialized();
        ts.init_error = entry->pipe->init_error();
        ts.input_url = c.input_url;
        ts.output_url = c.output_url;
        ts.preview_url = c.preview_url;
        ts.enable_preview = c.enable_preview;
        for (const auto& m : c.models) {
            ts.model_names.push_back(m.name);
            ts.model_types.push_back(m.type);
        }
        ts.stats_json = entry->pipe->stats().to_json();
        result.push_back(std::move(ts));
    }
    return result;
}

size_t PipelineManager::task_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pipelines_.size();
}

bool PipelineManager::add_model(const std::string& task_id, const ModelConfig& mcfg) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;
    bool ok = entry->pipe->add_model(mcfg);   // 模型加载在锁外
    if (ok) dirty_ = true;
    return ok;
}

bool PipelineManager::remove_model(const std::string& task_id, const std::string& model_name) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;
    bool ok = entry->pipe->remove_model(model_name);
    if (ok) dirty_ = true;
    return ok;
}

bool PipelineManager::update_model(const std::string& task_id, const std::string& model_name,
                                    const ModelConfig& mcfg) {
    auto entry = find_entry(task_id);
    if (!entry) return false;
    // 引擎加载（秒级）在 mtx_ 之外：热更新不再阻塞其它任务的控制面/查询
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) return false;
    bool ok = entry->pipe->update_model(model_name, mcfg);
    if (ok) dirty_ = true;
    return ok;
}

bool PipelineManager::update_task(const std::string& task_id, const TaskConfig& cfg, std::string* err) {
    auto entry = find_entry(task_id);
    if (!entry) {
        if (err) *err = "task not found";
        return false;
    }
    std::lock_guard<std::mutex> lk(entry->op_mtx);
    if (!find_entry(task_id)) {
        if (err) *err = "task not found";
        return false;
    }
    auto& pipe = *entry->pipe;
    // 需要先停止才能修改配置（op_mtx 保证与并发 start_task 互斥，不再有 TOCTOU）
    if (pipe.is_running()) {
        if (err) *err = "task must be stopped before updating config";
        return false;
    }

    // Diff 模型列表：移除旧有的、添加新增的（引擎加载在 mtx_ 之外）
    TaskConfig old_cfg = pipe.config();
    std::set<std::string> old_names, new_names;
    for (const auto& m : old_cfg.models) old_names.insert(m.name);
    for (const auto& m : cfg.models) new_names.insert(m.name);

    for (const auto& m : old_cfg.models) {
        if (!new_names.count(m.name)) {
            pipe.remove_model(m.name);
        }
    }
    for (const auto& m : cfg.models) {
        if (!old_names.count(m.name)) {
            pipe.add_model(m);
        }
    }

    bool ok = pipe.update_config(cfg);
    if (ok) dirty_ = true;
    return ok;
}

void PipelineManager::stop_all() {
    // 先摘除全部（映射清空，新操作找不到任何任务），再在 mtx_ 之外逐个停止/join。
    std::vector<std::pair<std::string, std::shared_ptr<TaskEntry>>> entries;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        entries.reserve(pipelines_.size());
        for (const auto& [id, e] : pipelines_) entries.emplace_back(id, e);
        pipelines_.clear();
        dirty_ = true;
    }
    for (auto& [id, entry] : entries) {
        if (!entry) continue;
        std::lock_guard<std::mutex> lk(entry->op_mtx);
        entry->pipe->stop();   // join 在 mtx_ 之外
    }
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
                            auto entry = std::make_shared<TaskEntry>();
                            entry->pipe = std::make_shared<Pipeline>(cfg, std::move(factory));
                            pipelines_[cfg.id] = std::move(entry);
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
            for (const auto& [id, entry] : pipelines_) {
                j["tasks"].push_back(task_config_to_json(entry->pipe->config()));
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

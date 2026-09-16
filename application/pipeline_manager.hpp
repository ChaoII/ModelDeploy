#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "config.hpp"
#include "pipeline.hpp"
#include "batched_detector.hpp"

/// 任务状态摘要
struct TaskStatus {
    std::string id;
    std::string name;
    bool running = false;
    bool initialized = false;
    std::string init_error;
    std::string input_url;
    std::string output_url;
    std::string preview_url;
    bool enable_preview = true;
    std::vector<std::string> model_names;
    std::vector<std::string> model_types;
    std::string stats_json;
};

/// 多任务管理器：线程安全 CRUD
class PipelineManager {
public:
    PipelineManager();
    ~PipelineManager();

    /// 创建并注册一个任务（不自动启动）
    bool create_task(const TaskConfig& cfg, std::string* err = nullptr);

    /// 设置某任务的检测回调（透传 Pipeline；Agent 用）
    bool set_detection_sink(const std::string& task_id, Pipeline::DetectionSink sink);

    /// 移除任务（如果正在运行则停止）
    bool remove_task(const std::string& task_id);

    /// 启动任务
    bool start_task(const std::string& task_id);

    /// 停止任务
    bool stop_task(const std::string& task_id);

    /// 获取任务配置
    bool get_task_config(const std::string& task_id, TaskConfig* cfg) const;

    /// 获取任务运行统计
    bool get_task_stats(const std::string& task_id, std::string* stats_json) const;

    /// 获取任务最新一帧的 JPEG
    bool get_task_jpeg(const std::string& task_id, std::vector<uint8_t>* jpeg, int quality = 80);

    /// 列出所有任务状态
    std::vector<TaskStatus> list_tasks() const;

    /// 获取任务数量
    size_t task_count() const;

    /// 动态管理模型
    bool add_model(const std::string& task_id, const ModelConfig& mcfg);
    bool remove_model(const std::string& task_id, const std::string& model_name);
    bool update_model(const std::string& task_id, const std::string& model_name,
                      const ModelConfig& mcfg);

    /// 更新任务核心配置（需先停止任务；不修改模型列表）
    bool update_task(const std::string& task_id, const TaskConfig& cfg, std::string* err = nullptr);

    /// 停止所有任务
    void stop_all();

    // ── 持久化 ──
    /// 从目录加载模型库与任务配置
    bool load_from_directory(const std::string& dir);
    /// 保存模型库与任务配置到目录
    bool save_to_directory(const std::string& dir);

    // ── 模型库管理 ──
    /// 添加模型到全局模型库
    bool add_model_to_library(const ModelConfig& mcfg);
    /// 从模型库删除模型
    bool remove_model_from_library(const std::string& name);
    /// 获取模型库列表
    std::vector<ModelConfig> list_models() const;
    /// 获取单个模型
    bool get_model(const std::string& name, ModelConfig* mcfg) const;
    /// 更新模型库中的模型
    bool update_model_in_library(const std::string& name, const ModelConfig& mcfg);

    /// 是否有未保存的变更
    bool is_dirty() const { return dirty_.load(); }
    void mark_clean() { dirty_ = false; }

    /// 模型工厂：供 Pipeline 创建 InferenceEngine 时通过 prototype cache 共享 ORT Session
    std::unique_ptr<InferenceEngine> create_engine(const ModelConfig& cfg);

    // ── 观测（测试/指标）：资源池存活条目数 ──
    /// 仍被使用者持有（未失效）的共享检测器数量
    size_t detector_pool_size() const;
    /// 模型原型缓存条目数
    size_t prototype_count() const;

private:
    // 任务条目：Pipeline 用 shared_ptr 持有，慢操作（模型加载/线程 join）在**不持 mtx_**
    // 的情况下进行（旧实现持全局 mtx_ 跨秒级加载/join → 所有控制面/调度/心跳一起阻塞，
    // 实测控制面请求超时；并发审计 HIGH）。取出条目后在锁外执行，shared_ptr 保证期间
    // remove_task 摘除映射也不会 UAF。
    struct TaskEntry {
        std::shared_ptr<Pipeline> pipe;
        // 串行化**同一任务**的慢操作与配置变更（启停/移除/热更新/改配置）：
        // 不同任务互不阻塞；读取路径（list_tasks/get_task_config 等）不持本锁，
        // 通过 Pipeline::config() 的 cfg_mtx_ 快照取得一致配置。
        mutable std::mutex op_mtx;
    };

    /// 按 id 取条目（持 mtx_ 拷贝 shared_ptr）；不存在返回 nullptr。
    /// 条目在调用方持有期间始终存活，可安全在锁外做慢操作。
    std::shared_ptr<TaskEntry> find_entry(const std::string& task_id) const;
    /// 取全部条目快照（持 mtx_ 拷贝后立即放锁）
    std::vector<std::pair<std::string, std::shared_ptr<TaskEntry>>> snapshot_entries() const;

    mutable std::mutex mtx_;   // 仅保护 pipelines_ / model_library_ 映射本身（短临界区）
    std::map<std::string, std::shared_ptr<TaskEntry>> pipelines_;
    std::vector<ModelConfig> model_library_;
    std::atomic<bool> dirty_{false};

    // 模型 prototype 缓存：只加载一次，后续 clone（共享 Runtime）
    struct ModelPrototype {
        std::string key;
        int type = 0; // 0=det, 1=face
        std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet> det;
        std::unique_ptr<modeldeploy::vision::face::Scrfd> face;
        uint64_t last_use = 0;   // LRU 时间戳（proto_tick_ 递增），用于淘汰无界增长
    };
    std::map<std::string, ModelPrototype> model_prototypes_;
    mutable std::mutex proto_mtx_;
    uint64_t proto_tick_ = 0;                       // 原型 LRU 时钟（仅 proto_mtx_ 内递增）
    // 原型缓存上限：不同配置反复热更新时防止无界驻留模型权重（克隆已独立共享 ORT session，
    // 淘汰原型不影响在用引擎，仅失去后续 clone 的共享）。
    static constexpr size_t kMaxPrototypes = 8;
    /// 在持 proto_mtx_ 时按 LRU 淘汰超限原型（保留 keep_key）
    void evict_prototypes_locked(const std::string& keep_key);

    // 共享批处理检测器池：同配置（路径/后端/设备/精度/尺寸/阈值）多路共享一个模型实例 +
    // AsyncModel 批推理 → GPU/CPU 高利用率、权重只驻留一份。
    // **弱引用**持有：旧实现用 shared_ptr 强引用且从不 erase，导致每次热更新换新 key 都
    // 永久驻留一个 BatchedDetector（含 ORT session + AsyncModel worker + ~8 线程）——
    // 实测 40 次热更新 +1.1GB / +319 线程，任务删除后仍不回落（并发审计 BLOCKER）。
    // weak_ptr 只在仍有 InferenceEngine 使用者时保活；最后一个使用者释放后检测器随即析构。
    std::map<std::string, std::weak_ptr<BatchedDetector>> det_pool_;
    mutable std::mutex det_pool_mtx_;
    std::shared_ptr<BatchedDetector> get_or_create_detector(const ModelConfig& cfg);
};

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "config.hpp"
#include "pipeline.hpp"
#include "stream_hub.hpp"

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
    std::vector<std::string> model_names;
    std::vector<std::string> model_types;
    std::string stats_json;
};

/// 多任务管理器：线程安全 CRUD
class PipelineManager {
public:
    PipelineManager() = default;
    ~PipelineManager();

    /// 创建并注册一个任务（不自动启动）
    bool create_task(const TaskConfig& cfg, std::string* err = nullptr);

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

private:
    mutable std::mutex mtx_;
    std::map<std::string, std::unique_ptr<Pipeline>> pipelines_;
    std::vector<ModelConfig> model_library_;  // 全局模型库
    std::atomic<bool> dirty_{false};
    StreamHub stream_hub_;  // 流共享中心
};

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "config.hpp"
#include "pipeline.hpp"

/// 任务状态摘要
struct TaskStatus {
    std::string id;
    std::string name;
    bool running = false;
    std::string input_url;
    std::string output_url;
    std::vector<std::string> model_names;
    std::vector<std::string> model_types;
};

/// 多任务管理器：线程安全 CRUD
class PipelineManager {
public:
    PipelineManager() = default;
    ~PipelineManager();

    /// 创建并注册一个任务（不自动启动）
    bool create_task(const TaskConfig& cfg);

    /// 移除任务（如果正在运行则停止）
    bool remove_task(const std::string& task_id);

    /// 启动任务
    bool start_task(const std::string& task_id);

    /// 停止任务
    bool stop_task(const std::string& task_id);

    /// 获取任务配置
    bool get_task_config(const std::string& task_id, TaskConfig* cfg) const;

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

    // ── 模型库管理 ──
    /// 添加模型到全局模型库
    bool add_model_to_library(const ModelConfig& mcfg);
    /// 从模型库删除模型
    bool remove_model_from_library(const std::string& name);
    /// 获取模型库列表
    std::vector<ModelConfig> list_models() const;
    /// 获取单个模型
    bool get_model(const std::string& name, ModelConfig* mcfg) const;

private:
    mutable std::mutex mtx_;
    std::map<std::string, std::unique_ptr<Pipeline>> pipelines_;
    std::vector<ModelConfig> model_library_;  // 全局模型库
};

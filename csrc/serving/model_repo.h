#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "core/md_decl.h"
#include "serving/config.h"

namespace modeldeploy::serving {

// 一次推理调用：输入 json → 输出 json，成功返回 true；失败返回 false 并填 err。
using InferFn = std::function<bool(const nlohmann::json& in, nlohmann::json* out, std::string* err)>;

struct MODELDEPLOY_CXX_EXPORT ModelHandle {
    std::string name;
    std::string version;
    bool ready = false;
    InferFn infer;
};

// 注入式句柄构造器：给定 name/version/模型目录，产出 InferFn（Task 3 提供真实实现）。
using HandleBuilder = std::function<InferFn(const std::string& name, const std::string& version,
                                            const std::string& model_dir)>;

// 模型仓库：扫描 repo/{name}/{version}/，登记地址、解析 latest、支持 scan 热更新。
class MODELDEPLOY_CXX_EXPORT ModelRepo {
public:
    // builder 可缺省（nullptr）→ 用内置占位 InferFn（返回 false + "not implemented (Task3)"）。
    explicit ModelRepo(const ServingConfig& cfg, HandleBuilder builder = nullptr,
                       std::string* err = nullptr);
    std::vector<std::string> scan();  // 返回本次新增/版本提升的模型名
    bool get(const std::string& name, const std::string& version, ModelHandle* out) const;
    std::vector<ModelHandle> list() const;

private:
    ModelHandle make_handle(const std::string& name, const std::string& version,
                            const std::string& dir) const;

    ServingConfig cfg_;
    HandleBuilder builder_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, ModelHandle> by_name_;              // name→当前激活版本句柄
    std::unordered_map<std::string, std::vector<std::string>> versions_;  // name→已登记版本
};

}  // namespace modeldeploy::serving

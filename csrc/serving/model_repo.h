#pragma once

#include <functional>
#include <memory>
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

struct ManifestModel;  // 完整定义见 serving/manifest.h

enum class ModelStatus { Unloaded, Loading, Ready, Failed };

struct MODELDEPLOY_CXX_EXPORT ModelHandle {
    std::string name;                  // manifest id
    std::string display;               // 前端显示名
    std::string version;               // 恒 "1"
    std::string type;                  // 前端渲染器选择：det/cls/seg/pose/ocr/face/lpr/obb/sem/depth
    std::vector<int> input_size;       // {w,h}
    std::vector<std::string> labels;   // 类别名（det/cls/seg/pose/obb 等有意义）
    std::string error;                 // status==Failed 时原因
    ModelStatus status = ModelStatus::Unloaded;
    InferFn infer;                     // 非空 ⇔ status==Ready
};

// 注入式句柄构造器：给定清单条目与资源根，产出 ModelHandle（Task 4 提供真实实现）。
using HandleBuilder = std::function<ModelHandle(const ManifestModel&, const std::string& base)>;

// 模型仓库：启动只读手写 manifest 目录元数据（status=Unloaded、infer 空），不实例化；
// load/unload 为单槽懒加载。目录仅由 manifest 提供，不再自动扫描。
class MODELDEPLOY_CXX_EXPORT ModelRepo {
public:
    // builder 可缺省（nullptr）→ 用内置占位 InferFn（恒返回 false + 未实现）。
    explicit ModelRepo(const ServingConfig& cfg, HandleBuilder builder = nullptr,
                       std::string* err = nullptr);
    ~ModelRepo();
    std::vector<std::string> scan();                 // 读 manifest，填充目录元数据；返回新增模型 id
    bool get(const std::string& name, ModelHandle* out) const;
    std::vector<ModelHandle> list() const;
    bool load(const std::string& name, std::string* err);   // 同步实例化，单槽，成功置 Ready
    bool unload(const std::string& name);                    // 卸下活跃模型，置 Unloaded

private:
    struct Impl;  // 持有 std::vector<ManifestModel>（需完整类型，故 pimpl 隐藏）

    ServingConfig cfg_;
    HandleBuilder builder_;
    std::string base_;
    std::string active_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, ModelHandle> by_name_;  // id→当前句柄（含目录元数据）
    std::unique_ptr<Impl> impl_;
};

}  // namespace modeldeploy::serving

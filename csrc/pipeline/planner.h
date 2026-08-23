#pragma once

#include "core/md_decl.h"
#include "csrc/pipeline/dag.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace modeldeploy::pipeline {

// 根据最小 DSL 自动布节点/边。YAGNI：只支持顺序 + 单级 fan-in/fan-out。
// 复杂图直接用 Dag::add_node/connect 手工搭。
class MODELDEPLOY_CXX_EXPORT Planner {
public:
    // 工厂：给定实例名创建 Node（端口 schema 由实现自行声明）
    using Factory = std::function<std::unique_ptr<Node>(const std::string& instance)>;

    void register_model(const std::string& name, Factory f);
    std::unique_ptr<Dag> build(const std::string& spec);

private:
    std::unordered_map<std::string, Factory> factories_;
};

} // namespace modeldeploy::pipeline

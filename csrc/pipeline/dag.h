#pragma once

#include "core/md_decl.h"
#include "csrc/pipeline/node.h"
#include "csrc/pipeline/edge.h"
#include <memory>
#include <string>
#include <vector>

namespace modeldeploy::pipeline {

// 有向无环图：注册节点 + 边，build() 校验并拓扑排序，execute() 拓扑序单线程执行。
class MODELDEPLOY_CXX_EXPORT Dag {
public:
    void add_node(std::unique_ptr<Node> node);
    Node* get_node(const std::string& name);

    // src_node.out_port -> dst_node.in_port；校验节点/端口存在且类型一致
    bool connect(const std::string& src_node, const std::string& src_port,
                 const std::string& dst_node, const std::string& dst_port);

    // 校验：无环 + 每个声明输入端口已连边或已种子 → 生成拓扑序；失败返回 false
    bool build();
    // 按拓扑序执行：前置清空各节点输出；每节点 run() 后把输出拷到下游输入
    bool execute();

    std::vector<std::string> execution_order() const { return order_; }
    const std::vector<Edge>& edges() const { return edges_; }

    // 独占所有 Node 所有权，不可拷贝（dllexport 下须显式声明以保证 move-only）
    Dag() = default;
    Dag(const Dag&) = delete;
    Dag& operator=(const Dag&) = delete;
    Dag(Dag&&) = default;
    Dag& operator=(Dag&&) = default;

private:
    Node* find_node(const std::string& name);

    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Edge> edges_;
    std::vector<std::string> order_;
    bool built_ = false;
};

} // namespace modeldeploy::pipeline

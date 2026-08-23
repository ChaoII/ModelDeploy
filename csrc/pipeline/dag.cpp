#include "csrc/pipeline/dag.h"
#include <deque>
#include <unordered_map>

namespace modeldeploy::pipeline {

Node* Dag::find_node(const std::string& name) {
    for (auto& n : nodes_) {
        if (n->name() == name) return n.get();
    }
    return nullptr;
}

void Dag::add_node(std::unique_ptr<Node> node) {
    nodes_.push_back(std::move(node));
    built_ = false;
}

Node* Dag::get_node(const std::string& name) { return find_node(name); }

bool Dag::connect(const std::string& src_node, const std::string& src_port,
                  const std::string& dst_node, const std::string& dst_port) {
    Node* s = find_node(src_node);
    Node* d = find_node(dst_node);
    if (!s || !d) return false;
    const Port* sp = nullptr;
    const Port* dp = nullptr;
    for (const auto& p : s->outputs()) if (p.name == src_port) { sp = &p; break; }
    for (const auto& p : d->inputs())  if (p.name == dst_port) { dp = &p; break; }
    if (!sp || !dp) return false;
    if (sp->type != dp->type) return false;  // 端口逻辑类型必须一致
    edges_.push_back(Edge{src_node, src_port, sp->type, dst_node, dst_port, dp->type});
    built_ = false;
    return true;
}

bool Dag::build() {
    if (nodes_.empty()) { built_ = true; order_.clear(); return true; }

    // 校验：每个声明的输入端口必须已连边 或 已种子（has_input）
    for (const auto& n : nodes_) {
        for (const auto& p : n->inputs()) {
            bool connected = false;
            for (const auto& e : edges_)
                if (e.dst_node == n->name() && e.dst_port == p.name) { connected = true; break; }
            if (!connected && !n->has_input(p.name)) return false;
        }
    }

    // Kahn 拓扑排序
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int> indeg;
    for (const auto& n : nodes_) { adj[n->name()] = {}; indeg[n->name()] = 0; }
    for (const auto& e : edges_) {
        adj[e.src_node].push_back(e.dst_node);
        indeg[e.dst_node]++;
    }
    std::deque<std::string> zero;
    for (const auto& n : nodes_) if (indeg[n->name()] == 0) zero.push_back(n->name());
    order_.clear();
    while (!zero.empty()) {
        std::string name = zero.front();
        zero.pop_front();
        order_.push_back(name);
        for (const auto& succ : adj[name])
            if (--indeg[succ] == 0) zero.push_back(succ);
    }
    built_ = (order_.size() == nodes_.size());  // 有环则 order 不全
    return built_;
}

bool Dag::execute() {
    if (!built_) return false;
    for (const auto& n : nodes_) n->clear_outputs();
    for (const auto& name : order_) {
        Node* n = find_node(name);
        if (!n) return false;
        if (!n->run()) return false;
        for (const auto& e : edges_) {
            if (e.src_node != name) continue;
            Node* d = find_node(e.dst_node);
            if (!d) return false;
            d->set_input(e.dst_port, n->get_output(e.src_port));
        }
    }
    return true;
}

} // namespace modeldeploy::pipeline

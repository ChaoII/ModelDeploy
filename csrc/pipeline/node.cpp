#include "csrc/pipeline/node.h"

namespace modeldeploy::pipeline {

Node::Node(std::string name, std::vector<Port> in, std::vector<Port> out)
    : name_(std::move(name)), in_(std::move(in)), out_(std::move(out)) {}

void Node::set_input(const std::string& port, std::any v) {
    in_data_[port] = std::move(v);
}

void Node::clear_outputs() {
    out_data_.clear();
}

bool Node::has_input(const std::string& port) const {
    auto it = in_data_.find(port);
    return it != in_data_.end() && it->second.has_value();
}

std::any Node::get_output(const std::string& port) const {
    auto it = out_data_.find(port);
    if (it == out_data_.end()) return {};
    return it->second;  // 拷贝：std::any 要求类型可拷贝（int/ImageData 均满足）
}

} // namespace modeldeploy::pipeline

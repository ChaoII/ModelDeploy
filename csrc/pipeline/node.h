#pragma once

#include "core/md_decl.h"
#include <any>
#include <string>
#include <unordered_map>
#include <vector>

namespace modeldeploy::pipeline {

// 端口：名称 + 逻辑类型名（"Image"/"DetectionResult"/"int" ...）
struct Port {
    std::string name;
    std::string type;
};

// 单一职责：声明输入/输出端口，run() 读取输入写入输出。数据用 std::any 承载。
class MODELDEPLOY_CXX_EXPORT Node {
public:
    Node(std::string name, std::vector<Port> in, std::vector<Port> out);
    virtual ~Node() = default;

    const std::string& name() const { return name_; }
    const std::vector<Port>& inputs() const { return in_; }
    const std::vector<Port>& outputs() const { return out_; }

    virtual bool run() = 0;

    // Dag 布线/种子使用：外部/上游写入输入；执行前置空输出；读取输出
    void set_input(const std::string& port, std::any v);
    void clear_outputs();
    bool has_input(const std::string& port) const;
    std::any get_output(const std::string& port) const;

protected:
    template <typename T>
    bool get_in(const std::string& port, T* out) {
        auto it = in_data_.find(port);
        if (it == in_data_.end() || !it->second.has_value()) return false;
        try {
            *out = std::any_cast<T>(it->second);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    template <typename T>
    void set_out(const std::string& port, const T& v) {
        out_data_[port] = std::any(v);
    }

private:
    std::string name_;
    std::vector<Port> in_, out_;
    std::unordered_map<std::string, std::any> in_data_, out_data_;
};

} // namespace modeldeploy::pipeline

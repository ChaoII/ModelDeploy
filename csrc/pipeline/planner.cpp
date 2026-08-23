#include "csrc/pipeline/planner.h"
#include <cctype>

namespace modeldeploy::pipeline {

void Planner::register_model(const std::string& name, Factory f) {
    factories_[name] = std::move(f);
}

namespace {
    std::string trim(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
        return s.substr(b, e - b);
    }

    // 拆顶层 "->"，得到各段
    std::vector<std::string> split_stages(const std::string& spec) {
        std::vector<std::string> out;
        size_t pos = 0;
        while (true) {
            auto arrow = spec.find("->", pos);
            if (arrow == std::string::npos) { out.push_back(trim(spec.substr(pos))); break; }
            out.push_back(trim(spec.substr(pos, arrow - pos)));
            pos = arrow + 2;
        }
        return out;
    }

    // 解析单段：可含单个名或 {A, B} 组
    std::vector<std::string> parse_stage(const std::string& seg) {
        std::vector<std::string> out;
        if (seg.empty()) return out;
        if (seg.front() == '{') {
            std::string cur;
            for (char c : seg) {
                if (c == '{' || c == '}') continue;
                if (c == ',') { out.push_back(trim(cur)); cur.clear(); }
                else cur += c;
            }
            if (!trim(cur).empty()) out.push_back(trim(cur));
        } else {
            out.push_back(seg);
        }
        return out;
    }

    // 把 model 名映射为唯一实例名（重复加 _N）
    std::string unique_instance(const std::string& model,
                                std::unordered_map<std::string, int>& counts) {
        int k = counts[model]++;
        return (k == 0) ? model : (model + "_" + std::to_string(k));
    }
} // namespace

std::unique_ptr<Dag> Planner::build(const std::string& spec) {
    std::vector<std::string> stages = split_stages(spec);
    if (stages.empty()) return nullptr;

    auto dag = std::make_unique<Dag>();
    std::unordered_map<std::string, int> counts;
    std::vector<std::vector<std::string>> groups;  // 每段的实例名

    for (const auto& st : stages) {
        std::vector<std::string> g = parse_stage(st);
        std::vector<std::string> instances;
        for (const auto& name : g) {
            auto it = factories_.find(name);
            if (it == factories_.end()) return nullptr;  // 未注册模型
            std::string inst = unique_instance(name, counts);
            dag->add_node(it->second(inst));
            instances.push_back(inst);
        }
        groups.push_back(std::move(instances));
    }

    for (size_t i = 0; i + 1 < groups.size(); ++i) {
        const auto& from = groups[i];
        const auto& to = groups[i + 1];
        if (from.size() == 1 && to.size() == 1) {
            if (!dag->connect(from[0], "out", to[0], "in")) return nullptr;
        } else if (from.size() > 1 && to.size() == 1) {
            // fan-in：源 k 连目标 in（k=0 为 in，k>=1 为 inN）
            for (size_t k = 0; k < from.size(); ++k) {
                std::string inport = (k == 0) ? std::string("in") : ("in" + std::to_string(k));
                if (!dag->connect(from[k], "out", to[0], inport)) return nullptr;
            }
        } else if (from.size() == 1 && to.size() > 1) {
            // fan-out：源 out -> 各目标 in
            for (const auto& t : to) {
                if (!dag->connect(from[0], "out", t, "in")) return nullptr;
            }
        } else {
            return nullptr;  // 多对多不支持（YAGNI）
        }
    }

    dag->build();  // 让 build 校验，调用方可再显式 build()
    return dag;
}

} // namespace modeldeploy::pipeline

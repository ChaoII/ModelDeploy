#include "serving/model_repo.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace modeldeploy::serving {

namespace fs = std::filesystem;

namespace {

// 内置占位 InferFn：真实推理由 Task 3 注入 HandleBuilder 提供。
InferFn placeholder_infer() {
    return [](const nlohmann::json&, nlohmann::json*, std::string* err) {
        if (err) *err = "not implemented (Task3)";
        return false;
    };
}

// 提取版本字符串中第一段连续数字。无数字返回 -1。
long long first_number(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && !std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i == s.size()) return -1;
    size_t j = i;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
    try {
        return std::stoll(s.substr(i, j - i));
    } catch (...) {
        return 0;
    }
}

// 数字优先比较：有数字则按数值（高者胜），同值或同为无数字 → 字典序；数字版本高于无数字版本。
bool version_higher(const std::string& a, const std::string& b) {
    const long long na = first_number(a);
    const long long nb = first_number(b);
    if (na >= 0 && nb >= 0) return na != nb ? na > nb : a > b;
    if (na >= 0) return true;
    if (nb >= 0) return false;
    return a > b;
}

}  // namespace

ModelRepo::ModelRepo(const ServingConfig& cfg, HandleBuilder builder, std::string* err)
    : cfg_(cfg) {
    (void)err;
    if (builder) {
        builder_ = std::move(builder);
    } else {
        builder_ = [](const std::string&, const std::string&, const std::string&) {
            return placeholder_infer();
        };
    }
}

ModelHandle ModelRepo::make_handle(const std::string& name, const std::string& version,
                                   const std::string& dir) const {
    ModelHandle h;
    h.name = name;
    h.version = version;
    h.ready = true;
    h.infer = builder_(name, version, dir);
    return h;
}

std::vector<std::string> ModelRepo::scan() {
    std::lock_guard<std::mutex> lock(mtx_);

    std::vector<std::string> changed;
    const fs::path root(cfg_.model_repo);
    std::error_code ec;
    if (!fs::exists(root, ec)) return changed;

    // 本次扫描：name → 已登记版本全集；name → 最高版本。
    std::unordered_map<std::string, std::vector<std::string>> scanned_versions;
    std::unordered_map<std::string, std::string> scanned_active;

    for (auto it = fs::directory_iterator(root, ec); it != fs::directory_iterator(); it.increment(ec)) {
        if (ec || !it->is_directory(ec)) continue;
        const std::string name = it->path().filename().string();

        std::vector<std::string> vers;
        for (auto vit = fs::directory_iterator(it->path(), ec); vit != fs::directory_iterator();
             vit.increment(ec)) {
            if (ec) continue;
            if (!vit->is_directory(ec)) continue;
            const std::string ver = vit->path().filename().string();
            std::error_code fec;
            if (fs::exists(vit->path() / "model.onnx", fec)) vers.push_back(ver);
        }
        if (vers.empty()) continue;  // 该 name 无任何合法版本

        std::sort(vers.begin(), vers.end(),
                  [](const std::string& a, const std::string& b) { return version_higher(a, b); });
        scanned_versions[name] = vers;
        scanned_active[name] = vers.front();
    }

    // 应用本次扫描：原子替换 by_name_ / versions_；移除已消失的 name。
    for (auto it = by_name_.begin(); it != by_name_.end();) {
        if (scanned_active.find(it->first) == scanned_active.end()) {
            versions_.erase(it->first);
            it = by_name_.erase(it);
        } else {
            ++it;
        }
    }

    for (const auto& kv : scanned_active) {
        const std::string& name = kv.first;
        const std::string& new_active = kv.second;

        auto prev = by_name_.find(name);
        const bool is_new = (prev == by_name_.end());
        const bool upgraded =
            !is_new && (prev->second.version.empty() || version_higher(new_active, prev->second.version));
        if (is_new || upgraded) changed.push_back(name);

        const fs::path dir = root / name / new_active;
        by_name_[name] = make_handle(name, new_active, dir.string());
        versions_[name] = scanned_versions.at(name);
    }

    return changed;
}

bool ModelRepo::get(const std::string& name, const std::string& version, ModelHandle* out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(mtx_);

    if (version.empty() || version == "latest") {
        auto it = by_name_.find(name);
        if (it == by_name_.end() || !it->second.ready) return false;
        *out = it->second;  // 值拷贝：旧持有者持有的 InferFn 不受热换影响
        return true;
    }

    auto vit = versions_.find(name);
    if (vit == versions_.end()) return false;
    if (std::find(vit->second.begin(), vit->second.end(), version) == vit->second.end()) return false;

    const fs::path dir = fs::path(cfg_.model_repo) / name / version;
    *out = make_handle(name, version, dir.string());
    return true;
}

std::vector<ModelHandle> ModelRepo::list() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ModelHandle> res;
    res.reserve(by_name_.size());
    for (const auto& kv : by_name_) res.push_back(kv.second);
    return res;
}

}  // namespace modeldeploy::serving

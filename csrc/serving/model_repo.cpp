#include "serving/model_repo.h"

#include "serving/manifest.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace modeldeploy::serving {

namespace fs = std::filesystem;

namespace {

// 内置占位 InferFn：真实推理由 HandleBuilder（Task 4）注入。
InferFn placeholder_infer() {
    return [](const nlohmann::json&, nlohmann::json*, std::string* err) {
        if (err) *err = "not implemented (lazy-load stub)";
        return false;
    };
}

// 读 manifest 顶层的资源根（base 字段），无则用 manifest 所在目录兜底。
std::string manifest_root_of(const std::string& manifest_path, const std::string& fallback) {
    try {
        std::ifstream f(manifest_path);
        nlohmann::json j;
        f >> j;
        if (j.contains("base") && j["base"].is_string()) {
            std::string b = j["base"].get<std::string>();
            if (!b.empty()) return b;
        }
    } catch (...) {
    }
    return fallback;
}

ModelHandle metadata_handle(const ManifestModel& m) {
    ModelHandle h;
    h.name = m.id;
    h.display = m.display;
    h.version = "1";
    h.type = m.type;
    h.input_size = m.input_size;
    h.labels = m.labels;
    h.error.clear();
    h.status = ModelStatus::Unloaded;
    return h;
}

}  // namespace

struct ModelRepo::Impl {
    std::vector<ManifestModel> manifest;
};

ModelRepo::~ModelRepo() = default;

ModelRepo::ModelRepo(const ServingConfig& cfg, HandleBuilder builder, std::string* err)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {
    (void)err;
    base_ = fs::path(cfg_.model_repo).parent_path().string();
    if (builder) {
        builder_ = std::move(builder);
    } else {
        builder_ = [](const ManifestModel&, const std::string&) {
            ModelHandle h;
            h.infer = placeholder_infer();
            return h;
        };
    }
}

std::vector<std::string> ModelRepo::scan() {
    std::lock_guard<std::mutex> lock(mtx_);

    base_ = manifest_root_of(cfg_.model_repo, fs::path(cfg_.model_repo).parent_path().string());

    std::vector<ManifestModel> fresh;
    std::string err;
    if (!load_manifest(cfg_.model_repo, base_, &fresh, &err)) {
        by_name_.clear();
        impl_->manifest.clear();
        return {};
    }
    impl_->manifest = std::move(fresh);

    std::vector<std::string> changed;
    for (const auto& mm : impl_->manifest) {
        const bool is_new = by_name_.find(mm.id) == by_name_.end();
        ModelHandle h = metadata_handle(mm);
        if (!is_new && active_ == mm.id) {
            auto& old = by_name_[mm.id];
            h.infer = old.infer;
            h.status = old.status;
            h.error = old.error;
        }
        by_name_[mm.id] = std::move(h);
        if (is_new) changed.push_back(mm.id);
    }
    // 移除 manifest 中已不存在的 id。
    for (auto it = by_name_.begin(); it != by_name_.end();) {
        const bool gone =
            std::find_if(impl_->manifest.begin(), impl_->manifest.end(),
                         [&](const ManifestModel& mm) { return mm.id == it->first; }) ==
            impl_->manifest.end();
        if (gone) {
            if (active_ == it->first) active_.clear();
            it = by_name_.erase(it);
        } else {
            ++it;
        }
    }
    return changed;
}

bool ModelRepo::get(const std::string& name, ModelHandle* out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return false;
    *out = it->second;
    return true;
}

std::vector<ModelHandle> ModelRepo::list() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ModelHandle> res;
    res.reserve(impl_->manifest.size());
    for (const auto& mm : impl_->manifest) {
        auto it = by_name_.find(mm.id);
        if (it != by_name_.end()) res.push_back(it->second);
    }
    return res;
}

const std::vector<ManifestModel>& ModelRepo::manifest() const { return impl_->manifest; }

bool ModelRepo::load(const std::string& name, std::string* err) {
    if (err) err->clear();
    ManifestModel mm;
    std::string base;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = by_name_.find(name);
        if (it == by_name_.end()) {
            if (err) *err = "model not found: " + name;
            return false;
        }
        // 在途保护：同一 id 正在后台构建 → no-op，返回当前（loading）状态，避免重复构建。
        if (it->second.status == ModelStatus::Loading) return true;
        // 单槽：卸下旧活跃模型。
        if (!active_.empty() && active_ != name) {
            auto prev = by_name_.find(active_);
            if (prev != by_name_.end()) {
                prev->second.infer = {};
                prev->second.status = ModelStatus::Unloaded;
                prev->second.error.clear();
            }
            active_.clear();
        }
        const ManifestModel* found = nullptr;
        for (const auto& mi : impl_->manifest)
            if (mi.id == name) { found = &mi; break; }
        if (!found) {
            if (err) *err = "model not in manifest: " + name;
            return false;
        }
        mm = *found;
        base = base_;
        it->second.status = ModelStatus::Loading;
        it->second.error.clear();
    }

    // 构建放在锁外：让 list()/get() 在构建期间可观察 loading 状态，不阻塞 HTTP 线程。
    ModelHandle built;
    try {
        built = builder_(mm, base);
    } catch (...) {
        built.status = ModelStatus::Failed;
        if (built.error.empty()) built.error = "construct threw";
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = by_name_.find(name);
        if (it == by_name_.end()) {
            if (err) *err = "model not found: " + name;
            return false;
        }
        // 构建期间被 unload/降级 → 丢弃本次结果。
        if (it->second.status != ModelStatus::Loading) return false;
        // 单槽：构建完成时仍保证至多一个活跃（可能被其它在途构建抢占）。
        if (!active_.empty() && active_ != name) {
            auto prev = by_name_.find(active_);
            if (prev != by_name_.end()) {
                prev->second.infer = {};
                prev->second.status = ModelStatus::Unloaded;
                prev->second.error.clear();
            }
            active_.clear();
        }
        if (built.infer) {
            it->second.infer = std::move(built.infer);
            it->second.status = ModelStatus::Ready;
            it->second.error.clear();
            active_ = name;
            return true;
        }
        it->second.status = ModelStatus::Failed;
        it->second.error = built.error.empty() ? "model failed to initialize" : built.error;
        if (err) *err = it->second.error;
        return false;
    }
}

bool ModelRepo::unload(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (active_ != name) return false;
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return false;
    it->second.infer = {};
    it->second.status = ModelStatus::Unloaded;
    it->second.error.clear();
    active_.clear();
    return true;
}

}  // namespace modeldeploy::serving

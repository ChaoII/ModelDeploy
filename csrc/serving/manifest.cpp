#include "serving/manifest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace modeldeploy::serving {

namespace fs = std::filesystem;

namespace {

// manifest 内的资源根：优先取 JSON 顶层 "base" 字段，否则用调用方传入的 base 兜底。
std::string manifest_root(const nlohmann::json& j, const std::string& fallback) {
    if (j.contains("base") && j["base"].is_string()) {
        std::string b = j["base"].get<std::string>();
        if (!b.empty()) return b;
    }
    return fallback;
}

std::string join_root(const std::string& root, const std::string& rel) {
    if (rel.empty()) return rel;
    const fs::path p(rel);
    if (p.is_absolute()) return rel;
    return (fs::path(root) / p).string();
}

// dict/labels 资产：仓库内已提交的文件（CWD 相对）优先，否则回退 base 拼接。
std::string resolve_asset(const std::string& root, const std::string& rel) {
    if (rel.empty()) return rel;
    const fs::path p(rel);
    if (p.is_absolute()) return rel;
    if (fs::exists(p)) return rel;
    return join_root(root, rel);
}

}  // namespace

bool load_manifest(const std::string& path, const std::string& base,
                   std::vector<ManifestModel>* out, std::string* err) {
    if (!out) {
        if (err) *err = "null output";
        return false;
    }
    out->clear();
    if (err) err->clear();

    std::ifstream f(path);
    if (!f) {
        if (err) *err = "cannot open manifest: " + path;
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (...) {
        if (err) *err = "invalid manifest JSON in " + path;
        return false;
    }
    if (!j.is_object()) {
        if (err) *err = "manifest root must be a JSON object";
        return false;
    }
    if (!j.contains("models") || !j["models"].is_array()) {
        if (err) *err = "manifest missing models[] array";
        return false;
    }

    const std::string root = manifest_root(j, base);
    for (const auto& e : j["models"]) {
        if (!e.is_object()) {
            if (err) *err = "manifest model entry must be an object";
            return false;
        }
        ManifestModel m;
        auto get_str = [&e](const char* key, std::string* v) {
            if (e.contains(key) && e[key].is_string()) *v = e[key].get<std::string>();
        };
        get_str("id", &m.id);
        get_str("display", &m.display);
        get_str("type", &m.type);
        get_str("desc", &m.desc);

        if (e.contains("input_size") && e["input_size"].is_array()) {
            m.input_size.clear();
            for (const auto& v : e["input_size"])
                if (v.is_number_integer()) m.input_size.push_back(v.get<int>());
        }
        if (m.input_size.empty()) m.input_size = {640, 640};

        if (e.contains("files") && e["files"].is_object()) {
            const auto& files = e["files"];
            auto g = [&](const char* key, std::string* v) {
                if (files.contains(key) && files[key].is_string())
                    *v = join_root(root, files[key].get<std::string>());
            };
            g("model", &m.model_f);
            g("rec", &m.rec_f);
            g("cls", &m.cls_f);
            if (files.contains("dict") && files["dict"].is_string())
                m.dict_f = resolve_asset(root, files["dict"].get<std::string>());
        }

        if (e.contains("labels")) {
            const auto& lab = e["labels"];
            if (lab.is_array()) {
                m.labels.clear();
                for (const auto& x : lab) if (x.is_string()) m.labels.push_back(x.get<std::string>());
            } else if (lab.is_string()) {
                std::ifstream lf(resolve_asset(root, lab.get<std::string>()));
                std::string line;
                while (std::getline(lf, line)) if (!line.empty()) m.labels.push_back(line);
            }
        }

        out->push_back(std::move(m));
    }
    return true;
}

}  // namespace modeldeploy::serving

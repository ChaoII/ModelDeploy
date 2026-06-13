#include "config.hpp"
#include <sstream>
#include <algorithm>
#include <cctype>

bool TaskConfig::validate(std::string* err) const {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    if (input_url.empty()) return fail("input_url is empty");
    if (output_url.empty()) return fail("output_url is empty");
    if (models.empty()) return fail("at least one model required");
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        if (m.name.empty()) return fail("model[" + std::to_string(i) + "].name is empty");
        if (m.path.empty()) return fail("model[" + std::to_string(i) + "].path is empty");
        if (m.input_size.size() != 2) return fail("model[" + std::to_string(i) + "].input_size must be [w,h]");
        if (m.roi.size() != 4) return fail("model[" + std::to_string(i) + "].roi must be [x,y,w,h]");
        if (m.interval < 1) return fail("model[" + std::to_string(i) + "].interval must be >= 1");
    }
    return true;
}

void TaskConfig::print() const {
    std::cout << "Task: " << name << " (" << id << ")\n"
              << "  input:  " << input_url << "\n"
              << "  output: " << output_url << "\n"
              << "  models: " << models.size() << "\n";
    for (const auto& m : models) {
        std::cout << "    - " << m.name << " [" << m.type << "] "
                  << m.path << "  backend=" << m.backend
                  << " device=" << m.device
                  << " threshold=" << m.confidence_threshold
                  << " interval=" << m.interval;
        if (m.roi[2] > 0 && m.roi[3] > 0)
            std::cout << " roi=[" << m.roi[0] << "," << m.roi[1]
                      << "," << m.roi[2] << "," << m.roi[3] << "]";
        std::cout << "\n";
    }
}

// ── TaskConfig ←→ json ──

json task_config_to_json(const TaskConfig& cfg) {
    json j;
    j["id"] = cfg.id;
    j["name"] = cfg.name;
    j["input_url"] = cfg.input_url;
    j["output_url"] = cfg.output_url;

    j["decoder"]["reconnect_delay_ms"] = cfg.decoder.reconnect_delay_ms;
    j["decoder"]["max_reconnects"] = cfg.decoder.max_reconnects;
    j["decoder"]["timeout_us"] = cfg.decoder.timeout_us;

    j["draw"]["show_label"] = cfg.draw.show_label;
    j["draw"]["show_score"] = cfg.draw.show_score;
    j["draw"]["font_path"] = cfg.draw.font_path;

    for (const auto& m : cfg.models) {
        json mo;
        mo["name"] = m.name;
        mo["type"] = m.type;
        mo["path"] = m.path;
        mo["backend"] = m.backend;
        mo["device"] = m.device;
        mo["confidence_threshold"] = m.confidence_threshold;
        mo["input_size"] = {m.input_size[0], m.input_size[1]};
        mo["roi"] = {m.roi[0], m.roi[1], m.roi[2], m.roi[3]};
        mo["interval"] = m.interval;
        if (!m.rec_path.empty()) mo["rec_path"] = m.rec_path;
        if (!m.labels.empty()) mo["labels"] = m.labels;
        j["models"].push_back(mo);
    }
    return j;
}

TaskConfig task_config_from_json(const json& j) {
    TaskConfig cfg;
    if (j.contains("id") && j["id"].is_string()) cfg.id = j["id"];
    if (j.contains("name") && j["name"].is_string()) cfg.name = j["name"];
    if (j.contains("input_url") && j["input_url"].is_string()) cfg.input_url = j["input_url"];
    if (j.contains("output_url") && j["output_url"].is_string()) cfg.output_url = j["output_url"];

    if (j.contains("decoder") && j["decoder"].is_object()) {
        auto& d = j["decoder"];
        if (d.contains("reconnect_delay_ms")) cfg.decoder.reconnect_delay_ms = d["reconnect_delay_ms"];
        if (d.contains("max_reconnects")) cfg.decoder.max_reconnects = d["max_reconnects"];
        if (d.contains("timeout_us")) cfg.decoder.timeout_us = d["timeout_us"];
    }

    if (j.contains("draw") && j["draw"].is_object()) {
        auto& d = j["draw"];
        if (d.contains("show_label")) cfg.draw.show_label = d["show_label"];
        if (d.contains("show_score")) cfg.draw.show_score = d["show_score"];
        if (d.contains("font_path")) cfg.draw.font_path = d["font_path"];
    }

    if (j.contains("models") && j["models"].is_array()) {
        for (const auto& mo : j["models"]) {
            ModelConfig m;
            if (mo.contains("name")) m.name = mo["name"];
            if (mo.contains("type")) m.type = mo["type"];
            if (mo.contains("path")) m.path = mo["path"];
            if (mo.contains("rec_path")) m.rec_path = mo["rec_path"];
            if (mo.contains("backend")) m.backend = mo["backend"];
            if (mo.contains("device")) m.device = mo["device"];
            if (mo.contains("confidence_threshold")) m.confidence_threshold = mo["confidence_threshold"];
            if (mo.contains("input_size") && mo["input_size"].is_array() && mo["input_size"].size() >= 2)
                m.input_size = {mo["input_size"][0], mo["input_size"][1]};
            if (mo.contains("roi") && mo["roi"].is_array() && mo["roi"].size() >= 4)
                m.roi = {mo["roi"][0], mo["roi"][1], mo["roi"][2], mo["roi"][3]};
            if (mo.contains("interval")) m.interval = mo["interval"];
            if (mo.contains("labels") && mo["labels"].is_array())
                m.labels = mo["labels"].get<std::vector<std::string>>();
            cfg.models.push_back(m);
        }
    }
    return cfg;
}

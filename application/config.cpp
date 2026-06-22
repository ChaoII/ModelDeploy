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
    if (enable_preview && output_url.empty()) return fail("output_url is empty when enable_preview=true");
    if (models.empty()) return fail("at least one model required");
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        if (m.name.empty()) return fail("model[" + std::to_string(i) + "].name is empty");
        if (m.path.empty()) return fail("model[" + std::to_string(i) + "].path is empty");
        if (m.input_size.size() != 2) return fail("model[" + std::to_string(i) + "].input_size must be [w,h]");
        if (m.roi.size() != 4) return fail("model[" + std::to_string(i) + "].roi must be [x,y,w,h]");
        if (m.interval < 1) return fail("model[" + std::to_string(i) + "].interval must be >= 1");
    }
    if (enable_preview) {
        if (encoder.fps < 0 || encoder.fps > 120) return fail("encoder.fps out of range (0-120)");
        if (encoder.bitrate_kbps <= 0) return fail("encoder.bitrate_kbps must be > 0");
    }
    return true;
}

void TaskConfig::print() const {
    std::cout << "Task: " << name << " (" << id << ")\n"
              << "  input:  " << input_url << "\n"
              << "  output: " << (enable_preview ? output_url : "(no preview)") << "\n"
              << "  preview: " << preview_url << "\n"
              << "  enable_preview: " << (enable_preview ? "true" : "false") << "\n"
              << "  encoder: " << encoder.codec << "/" << encoder.format
              << " " << encoder.fps << "fps " << encoder.bitrate_kbps << "kbps gop=" << encoder.gop << "\n"
              << "  decoder: " << decoder.hw_accel << " transport=" << decoder.rtsp_transport << "\n"
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
    j["preview_url"] = cfg.preview_url;
    j["enable_preview"] = cfg.enable_preview;

    j["decoder"]["reconnect_delay_ms"] = cfg.decoder.reconnect_delay_ms;
    j["decoder"]["max_reconnects"] = cfg.decoder.max_reconnects;
    j["decoder"]["timeout_us"] = cfg.decoder.timeout_us;
    j["decoder"]["rtsp_transport"] = cfg.decoder.rtsp_transport;
    j["decoder"]["hw_accel"] = cfg.decoder.hw_accel;

    j["encoder"]["fps"] = cfg.encoder.fps;
    j["encoder"]["bitrate_kbps"] = cfg.encoder.bitrate_kbps;
    j["encoder"]["gop"] = cfg.encoder.gop;
    j["encoder"]["codec"] = cfg.encoder.codec;
    j["encoder"]["preset"] = cfg.encoder.preset;
    j["encoder"]["tune"] = cfg.encoder.tune;
    j["encoder"]["format"] = cfg.encoder.format;
    j["encoder"]["max_b_frames"] = cfg.encoder.max_b_frames;
    j["encoder"]["low_latency"] = cfg.encoder.low_latency;

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
    if (j.contains("preview_url") && j["preview_url"].is_string()) cfg.preview_url = j["preview_url"];
    if (j.contains("enable_preview") && j["enable_preview"].is_boolean()) cfg.enable_preview = j["enable_preview"];
    else if (j.contains("enable_preview") && j["enable_preview"].is_number()) cfg.enable_preview = j["enable_preview"].get<int>() != 0;

    if (j.contains("decoder") && j["decoder"].is_object()) {
        auto& d = j["decoder"];
        if (d.contains("reconnect_delay_ms")) cfg.decoder.reconnect_delay_ms = d["reconnect_delay_ms"];
        if (d.contains("max_reconnects")) cfg.decoder.max_reconnects = d["max_reconnects"];
        if (d.contains("timeout_us")) cfg.decoder.timeout_us = d["timeout_us"];
        if (d.contains("rtsp_transport") && d["rtsp_transport"].is_string()) cfg.decoder.rtsp_transport = d["rtsp_transport"];
        if (d.contains("hw_accel") && d["hw_accel"].is_string()) cfg.decoder.hw_accel = d["hw_accel"];
    }

    if (j.contains("encoder") && j["encoder"].is_object()) {
        auto& e = j["encoder"];
        if (e.contains("fps")) cfg.encoder.fps = e["fps"];
        if (e.contains("bitrate_kbps")) cfg.encoder.bitrate_kbps = e["bitrate_kbps"];
        if (e.contains("gop")) cfg.encoder.gop = e["gop"];
        if (e.contains("codec") && e["codec"].is_string()) cfg.encoder.codec = e["codec"];
        if (e.contains("preset") && e["preset"].is_string()) cfg.encoder.preset = e["preset"];
        if (e.contains("tune") && e["tune"].is_string()) cfg.encoder.tune = e["tune"];
        if (e.contains("format") && e["format"].is_string()) cfg.encoder.format = e["format"];
        if (e.contains("max_b_frames")) cfg.encoder.max_b_frames = e["max_b_frames"];
        if (e.contains("low_latency")) cfg.encoder.low_latency = e["low_latency"];
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

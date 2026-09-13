#include "config_adapter.hpp"
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

std::string normalize_model_type(const std::string& type) {
    if (type == "det" || type == "detection") return "detection";
    if (type == "cls" || type == "classification") return "classification";
    if (type == "face" || type == "face_detection") return "face_detection";
    return type;
}

std::vector<float> ConfigAdapter::polygon_to_norm_rect(const json& roi) {
    if (!roi.is_array() || roi.size() < 2) return {};  // 允许矩形两角点 [x0,y0],[x1,y1]
    float xmin = 1.f, ymin = 1.f, xmax = 0.f, ymax = 0.f;
    for (const auto& p : roi) {
        if (!p.is_array() || p.size() < 2 || !p[0].is_number() || !p[1].is_number()) return {};
        // 不可信输入：归一化坐标钳制到 [0,1]
        const float x = std::clamp(p[0].get<float>(), 0.f, 1.f);
        const float y = std::clamp(p[1].get<float>(), 0.f, 1.f);
        xmin = std::min(xmin, x); ymin = std::min(ymin, y);
        xmax = std::max(xmax, x); ymax = std::max(ymax, y);
    }
    if (xmax <= xmin || ymax <= ymin) return {};
    return {xmin, ymin, xmax - xmin, ymax - ymin};
}

bool ConfigAdapter::from_json(const json& j, AdaptedTask* out, std::string* err) const {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!out) return fail("null output");
    if (!j.is_object()) return fail("task config must be a json object");

    // task_id
    if (j.contains("task_id") && j["task_id"].is_number_integer()) {
        out->sdk.id = std::to_string(j["task_id"].get<int64_t>());
        out->camera.id = 0;  // overwritten below if camera.id present
    } else if (j.contains("task_id") && j["task_id"].is_string()) {
        out->sdk.id = j["task_id"].get<std::string>();
    } else {
        return fail("task_id required");
    }

    // camera
    if (!j.contains("camera") || !j["camera"].is_object()) return fail("camera required");
    const auto& cam = j["camera"];
    if (cam.contains("id") && cam["id"].is_number_integer()) out->camera.id = cam["id"].get<int>();
    if (cam.contains("name") && cam["name"].is_string()) out->camera.name = cam["name"].get<std::string>();
    if (cam.contains("url") && cam["url"].is_string()) out->camera.url = cam["url"].get<std::string>();
    if (cam.contains("transport") && cam["transport"].is_string()) out->camera.transport = cam["transport"].get<std::string>();
    if (out->camera.url.empty()) return fail("camera.url required");
    out->sdk.input_url = out->camera.url;
    out->sdk.name = out->camera.name;
    out->sdk.decoder.rtsp_transport = out->camera.transport;

    // decoder / encoder 透传
    if (j.contains("decoder") && j["decoder"].is_object()) {
        const auto& d = j["decoder"];
        if (d.contains("hw_accel") && d["hw_accel"].is_string()) out->sdk.decoder.hw_accel = d["hw_accel"];
        if (d.contains("device_only") && d["device_only"].is_boolean()) out->sdk.decoder.device_only = d["device_only"];
        if (d.contains("rtsp_transport") && d["rtsp_transport"].is_string()) out->sdk.decoder.rtsp_transport = d["rtsp_transport"];
    }
    if (j.contains("encoder") && j["encoder"].is_object()) {
        const auto& e = j["encoder"];
        if (e.contains("codec") && e["codec"].is_string()) out->sdk.encoder.codec = e["codec"];
        if (e.contains("format") && e["format"].is_string()) out->sdk.encoder.format = e["format"];
        if (e.contains("bitrate_kbps") && e["bitrate_kbps"].is_number_integer()) out->sdk.encoder.bitrate_kbps = e["bitrate_kbps"];
    }

    // preview
    out->sdk.enable_preview = false;
    if (j.contains("preview") && j["preview"].is_object()) {
        const auto& p = j["preview"];
        if (p.contains("enabled") && p["enabled"].is_boolean()) out->sdk.enable_preview = p["enabled"].get<bool>();
        if (p.contains("format") && p["format"].is_string()) out->sdk.encoder.format = p["format"].get<std::string>();
    }
    if (out->sdk.enable_preview) {
        // AIStation 未下发 output_url；Agent 生成 FLV 输出（主流程任务 9 接线）
        out->sdk.output_url = "aistation/" + out->sdk.id + ".flv";
        out->sdk.preview_url = out->sdk.output_url;
    }

    // roi
    const std::vector<float> rect = j.contains("roi") ? polygon_to_norm_rect(j["roi"]) : std::vector<float>{};

    // models
    out->sdk.models.clear();
    if (!j.contains("models") || !j["models"].is_array() || j["models"].empty())
        return fail("at least one model required");
    for (const auto& mo : j["models"]) {
        if (!mo.is_object()) return fail("model entry must be object");
        ModelConfig m;
        if (mo.contains("name") && mo["name"].is_string()) m.name = mo["name"];
        if (mo.contains("type") && mo["type"].is_string()) m.type = normalize_model_type(mo["type"]);
        if (mo.contains("backend") && mo["backend"].is_string()) m.backend = mo["backend"];
        if (mo.contains("device") && mo["device"].is_string()) m.device = mo["device"];
        if (mo.contains("confidence_threshold") && mo["confidence_threshold"].is_number())
            m.confidence_threshold = mo["confidence_threshold"].get<float>();
        if (mo.contains("input_size") && mo["input_size"].is_array() && mo["input_size"].size() >= 2)
            m.input_size = {mo["input_size"][0], mo["input_size"][1]};
        if (mo.contains("labels") && mo["labels"].is_array())
            m.labels = mo["labels"].get<std::vector<std::string>>();
        m.roi_norm = rect;
        // path 优先；无 path 时仅接受本地 url（远端拉取见 Task 9）
        if (mo.contains("path") && mo["path"].is_string()) {
            m.path = mo["path"];
        } else if (mo.contains("url") && mo["url"].is_string()) {
            const std::string url = mo["url"];
            if (url.rfind("file://", 0) == 0) m.path = url.substr(7);
            else if (url.find("://") == std::string::npos) m.path = url;
            else return fail("remote model url requires model_fetcher: " + url);
        }
        if (m.name.empty()) return fail("model.name required");
        if (m.path.empty()) return fail("model.path/url required for " + m.name);
        out->sdk.models.push_back(std::move(m));
    }

    // events
    out->alarm_interval_sec = j.value("alarm_interval_sec", 0);
    if (j.contains("algorithm_type") && j["algorithm_type"].is_string())
        out->algorithm_type = j["algorithm_type"].get<std::string>();
    if (j.contains("tenant") && j["tenant"].is_string()) out->tenant = j["tenant"].get<std::string>();

    if (j.contains("events") && j["events"].is_object()) {
        const auto& ev = j["events"];
        if (ev.contains("transport") && ev["transport"].is_string()) out->events.transport = ev["transport"];
        if (ev.contains("mqtt") && ev["mqtt"].is_object()) {
            const auto& m = ev["mqtt"];
            if (m.contains("broker") && m["broker"].is_string()) out->events.mqtt_broker = m["broker"];
            if (m.contains("topic") && m["topic"].is_string()) out->events.mqtt_topic = m["topic"];
            if (m.contains("qos") && m["qos"].is_number_integer()) out->events.mqtt_qos = m["qos"];
            if (m.contains("client_id") && m["client_id"].is_string()) out->events.mqtt_client_id = m["client_id"];
            if (m.contains("username") && m["username"].is_string()) out->events.mqtt_username = m["username"];
            if (m.contains("password") && m["password"].is_string()) out->events.mqtt_password = m["password"];
        }
        if (ev.contains("http") && ev["http"].is_object()) {
            const auto& h = ev["http"];
            if (h.contains("url") && h["url"].is_string()) out->events.http_url = h["url"];
            if (h.contains("token") && h["token"].is_string()) out->events.http_token = h["token"];
        }
        if (ev.contains("buffer") && ev["buffer"].is_object()) {
            const auto& b = ev["buffer"];
            if (b.contains("dir") && b["dir"].is_string()) out->events.buffer_dir = b["dir"];
            if (b.contains("max_mb") && b["max_mb"].is_number_integer()) out->events.buffer_max_mb = b["max_mb"];
        }
    }
    return true;
}

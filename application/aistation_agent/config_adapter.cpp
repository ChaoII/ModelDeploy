#include "config_adapter.hpp"
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

std::string normalize_model_type(const std::string& type) {
    if (type == "det" || type == "detection") return "detection";
    if (type == "cls" || type == "classification") return "classification";
    if (type == "face" || type == "face_detection") return "face_detection";
    if (type == "ped_attr" || type == "pedestrian_attribute") return "pedestrian_attribute";
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

    // preview：仅真实流式为 muxer 时启用编码预览；"snapshot"/未知格式保持关闭
    out->sdk.enable_preview = false;
    if (j.contains("preview") && j["preview"].is_object()) {
        const auto& p = j["preview"];
        bool enabled = false;
        if (p.contains("enabled") && p["enabled"].is_boolean()) enabled = p["enabled"].get<bool>();
        std::string fmt;
        if (p.contains("format") && p["format"].is_string()) fmt = p["format"].get<std::string>();
        if (enabled && (fmt == "flv" || fmt == "rtmp")) {
            out->sdk.enable_preview = true;
            out->sdk.encoder.format = fmt;
        }
    }
    if (out->sdk.enable_preview) {
        // AIStation 未下发 output_url；Agent 生成 FLV 输出（主流程任务 9 接线）
        out->sdk.output_url = "aistation/" + out->sdk.id + ".flv";
        out->sdk.preview_url = out->sdk.output_url;
    }

    // roi
    const std::vector<float> rect = j.contains("roi") ? polygon_to_norm_rect(j["roi"]) : std::vector<float>{};
    out->roi_polygon = aistation::parse_roi_polygon(j.contains("roi") ? j["roi"] : nlohmann::json::array());
    out->schedule = aistation::parse_schedule(j.contains("schedule") ? j["schedule"] : nlohmann::json::object());

    // models
    out->sdk.models.clear();
    if (!j.contains("models") || !j["models"].is_array() || j["models"].empty())
        return fail("at least one model required");
    for (const auto& mo : j["models"]) {
        ModelConfig m;
        if (!parse_one_model(mo, rect, &m, err)) return false;
        out->sdk.models.push_back(std::move(m));
    }

    // tracking：顶层跟踪配置（默认关闭）
    if (j.contains("tracking") && j["tracking"].is_object()) {
        const auto& tk = j["tracking"];
        if (tk.contains("enabled") && tk["enabled"].is_boolean()) out->tracking.enabled = tk["enabled"].get<bool>();
        if (tk.contains("algorithm") && tk["algorithm"].is_string()) out->tracking.algorithm = tk["algorithm"].get<std::string>();
    }
    // 单一事实源：Agent 用 t.sdk（TaskConfig）建 Pipeline（PipelineManager::create_task(t.sdk)），
    // 故必须把顶层 tracking 同步写入 sdk.tracking，否则跟踪配置在 Pipeline 侧丢失
    out->sdk.tracking = out->tracking;

    // events
    out->alarm_interval_sec = j.value("alarm_interval_sec", 0);
    out->heartbeat_sec = j.value("heartbeat_sec", 5);
    if (j.contains("algorithm_type") && j["algorithm_type"].is_string())
        out->algorithm_type = j["algorithm_type"].get<std::string>();
    if (j.contains("tenant") && j["tenant"].is_string()) out->tenant = j["tenant"].get<std::string>();

    if (j.contains("events") && j["events"].is_object()) {
        const auto& ev = j["events"];
        if (ev.contains("transport") && ev["transport"].is_string()) out->events.transport = ev["transport"];
        if (ev.contains("mqtt") && ev["mqtt"].is_object()) {
            const auto& m = ev["mqtt"];
            if (m.contains("broker") && m["broker"].is_string())
                out->events.mqtt_broker = aistation::normalize_broker_scheme(m["broker"].get<std::string>());
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
        if (ev.contains("snapshot") && ev["snapshot"].is_object()) {
            const auto& s = ev["snapshot"];
            if (s.contains("enabled") && s["enabled"].is_boolean()) out->snapshot.enabled = s["enabled"].get<bool>();
            if (s.contains("inline") && s["inline"].is_boolean()) out->snapshot.inline_snapshot = s["inline"].get<bool>();
            if (s.contains("quality") && s["quality"].is_number_integer()) out->snapshot.quality = s["quality"].get<int>();
            if (s.contains("max_width") && s["max_width"].is_number_integer()) out->snapshot.max_width = s["max_width"].get<int>();
        }
    }
    return true;
}

// models[] 单个条目的字段映射：from_json 与 parse_model 共用同一实现，避免字段漂移。
bool ConfigAdapter::parse_one_model(const json& mo, const std::vector<float>& roi_norm,
                                    ModelConfig* out, std::string* err) const {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!out) return fail("null output");
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
    if (mo.contains("attributes") && mo["attributes"].is_array())
        m.attributes = mo["attributes"].get<std::vector<std::string>>();
    if (mo.contains("password") && mo["password"].is_string())
        m.password = mo["password"].get<std::string>();
    m.roi_norm = roi_norm;
    // 多模型 pipeline：*_url → 本地路径（本地直用；远端经 ModelFetcher）
    auto resolve_path = [&](const std::string& key, std::string* dst) -> bool {
        if (!mo.contains(key) || !mo[key].is_string()) return true;
        const std::string url = mo[key].get<std::string>();
        if (url.rfind("file://", 0) == 0 || url.find("://") == std::string::npos) {
            *dst = url.rfind("file://", 0) == 0 ? url.substr(7) : url;
            return true;
        }
        if (!fetcher_) return fail("remote model url requires model_fetcher: " + url);
        std::string local, ferr;
        if (!fetcher_->fetch(url, &local, &ferr)) return fail("model fetch failed: " + ferr);
        *dst = local;
        return true;
    };
    if (m.type == "ocr") {
        // OCR pipeline：det_url→path、cls_url→cls_path、rec_url→rec_path、dict_url→dict_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("cls_url", &m.cls_path)) return false;
        if (!resolve_path("rec_url", &m.rec_path)) return false;
        if (!resolve_path("dict_url", &m.dict_path)) return false;
    } else if (m.type == "lpr") {
        // LPR 车牌 pipeline：det_url→path、rec_url→rec_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("rec_url", &m.rec_path)) return false;
    } else {
        // pedestrian_attribute pipeline：det_url→path、cls_url→rec_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("cls_url", &m.rec_path)) return false;
    }
    // 单模型：path 优先；无 path 时本地/远端 url 均支持（远端经 ModelFetcher）
    if (m.path.empty()) {
        if (mo.contains("path") && mo["path"].is_string()) {
            m.path = mo["path"];
        } else if (mo.contains("url") && mo["url"].is_string()) {
            const std::string url = mo["url"];
            if (url.rfind("file://", 0) == 0 || url.find("://") == std::string::npos) {
                m.path = url.rfind("file://", 0) == 0 ? url.substr(7) : url;
            } else if (fetcher_) {
                std::string local, ferr;
                if (!fetcher_->fetch(url, &local, &ferr))
                    return fail("model fetch failed for " + m.name + ": " + ferr);
                m.path = local;
            } else {
                return fail("remote model url requires model_fetcher: " + url);
            }
        }
    }
    if (m.name.empty()) return fail("model.name required");
    if (m.path.empty()) return fail("model.path/url required for " + m.name);
    *out = std::move(m);
    return true;
}

bool ConfigAdapter::parse_model(const json& j, ModelConfig* out, std::string* err,
                                const std::vector<float>& roi_norm) const {
    return parse_one_model(j, roi_norm, out, err);
}

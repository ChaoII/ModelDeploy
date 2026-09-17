#include "config_adapter.hpp"
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

std::string normalize_model_type(const std::string& type) {
    if (type == "det" || type == "detection") return "detection";
    if (type == "cls" || type == "classification") return "classification";
    if (type == "face" || type == "face_detection") return "face_detection";
    if (type == "ped_attr" || type == "pedestrian_attribute") return "pedestrian_attribute";
    // 旋转目标（场景目录 OBB_DET 声明 type="obb"）
    if (type == "obb" || type == "obb_det" || type == "rotated_detection") return "obb";
    // 实例分割（场景目录 I_SEG 声明 type="iseg"；云端编排关键词映射可能给出 "seg"）
    if (type == "iseg" || type == "seg" || type == "instance_seg" ||
        type == "instance_segmentation")
        return "iseg";
    // 人体姿态关键点（场景目录 POSE/FALL/CLIMB/SMOKE_PHONE/HAND_GESTURE 声明 type="pose"；
    // 兼容 SDK 侧可能的其它拼写：pose_estimation / keypoint(s) / human_pose / yolo_pose）
    if (type == "pose" || type == "pose_estimation" || type == "keypoint" ||
        type == "keypoints" || type == "keypoint_detection" || type == "human_pose" ||
        type == "yolo_pose")
        return "pose";
    // 人脸属性（场景目录 FACE_ATTR 声明 type="face_attr"；性别/年龄）
    if (type == "face_attr" || type == "face_attribute" || type == "face_attributes" ||
        type == "face_attr_pipeline")
        return "face_attr";
    // 人脸活体/防伪（场景目录 FACE_ANTISPOOF 声明 type="face_as"）
    if (type == "face_as" || type == "face_antispoof" || type == "antispoof" ||
        type == "anti_spoof" || type == "face_liveness" || type == "liveness" ||
        type == "face_as_pipeline")
        return "face_as";
    // 单目深度估计（场景目录 DEPTH_SAFE 声明 type="depth"）
    if (type == "depth" || type == "depth_estimation" || type == "mono_depth" ||
        type == "yolo_depth" || type == "distance")
        return "depth";
    // 语义分割（场景目录 SEM_AREA 声明 type="sem"）：整帧占比 → attributes{类别名: 占比}
    if (type == "sem" || type == "semantic_segmentation" || type == "semantic_seg" ||
        type == "sem_seg" || type == "yolo_sem" || type == "sem_pipeline")
        return "sem";
    // 人脸关键点（场景目录 FACE_LANDMARK 声明 type="face_landmark"，106 点）
    // 与 pose 共用事件字段 objects[].keypoints（归一化 [x,y,score]）
    if (type == "face_landmark" || type == "face_landmarks" || type == "face_lmk" ||
        type == "landmark" || type == "landmarks" || type == "face_keypoint" ||
        type == "face_keypoints" || type == "2d106" || type == "face_landmark_pipeline")
        return "face_landmark";
    // 文档/表格（场景目录 DOC_TABLE 的 pipeline 为 [ocr, cls]；本族直连 OCR 导出文本）
    if (type == "doc" || type == "doc_table" || type == "document" || type == "document_table" ||
        type == "table" || type == "table_structure" || type == "doc_structure" ||
        type == "ppstructure" || type == "layout")
        return "doc";
    // 条码/二维码（场景目录 BARCODE 的 pipeline 为 [ocr]；本族用原生 ZXing 解码，无需权重）
    if (type == "barcode" || type == "barcodes" || type == "qr" || type == "qrcode" ||
        type == "qr_code" || type == "bar_code" || type == "code_reader" ||
        type == "barcode_pipeline")
        return "barcode";
    // 交互式分割（场景目录 SAM_SEG 的 pipeline 声明为 [iseg]，模型为 FastSAM-s）
    if (type == "sam" || type == "fastsam" || type == "fast_sam" || type == "fastsam_s" ||
        type == "fast_sam_s" || type == "sam_seg" || type == "segment_anything" ||
        type == "sam_pipeline")
        return "sam";
    // 人脸识别/人脸嵌入（场景目录 FACE_REC/STRANGER 的 pipeline 为 [face_detection, face_rec]）
    if (type == "face_rec" || type == "face_recognition" || type == "face_embedding" ||
        type == "face_embed" || type == "face_feature" || type == "face_id" ||
        type == "arcface" || type == "face_rec_pipeline")
        return "face_rec";
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

    // 任务级提示点（SAM_SEG 场景参数 "prompt_point"，归一化 [x,y]）：模型条目未带时回退使用
    std::vector<float> task_prompt;
    if (j.contains("prompt_point") && j["prompt_point"].is_array() && j["prompt_point"].size() >= 2 &&
        j["prompt_point"][0].is_number() && j["prompt_point"][1].is_number()) {
        task_prompt = {std::clamp(j["prompt_point"][0].get<float>(), 0.f, 1.f),
                       std::clamp(j["prompt_point"][1].get<float>(), 0.f, 1.f)};
    } else if (j.contains("prompt_point") && j["prompt_point"].is_object() &&
               j["prompt_point"].contains("x") && j["prompt_point"].contains("y") &&
               j["prompt_point"]["x"].is_number() && j["prompt_point"]["y"].is_number()) {
        task_prompt = {std::clamp(j["prompt_point"]["x"].get<float>(), 0.f, 1.f),
                       std::clamp(j["prompt_point"]["y"].get<float>(), 0.f, 1.f)};
    }

    // 人脸检测器复用：目录 FACE_REC/STRANGER 的 pipeline 为 [face_detection, face_rec]、
    // FACE_LANDMARK 为 [face_detection, face_landmark] 两条独立模型，而 Agent 的
    // face_rec/face_landmark 是「检测 + 次级模型」复合模型。若这些条目未显式给出 det_url/
    // det_path，则复用同一任务中 face_detection 条目的模型路径作为检测器（SCRFD）。
    std::string shared_face_det;
    if (j.contains("models") && j["models"].is_array()) {
        for (const auto& mo : j["models"]) {
            if (!mo.is_object() || !mo.contains("type") || !mo["type"].is_string()) continue;
            if (normalize_model_type(mo["type"].get<std::string>()) != "face_detection") continue;
            if (mo.contains("url") && mo["url"].is_string())
                shared_face_det = mo["url"].get<std::string>();
            else if (mo.contains("path") && mo["path"].is_string())
                shared_face_det = mo["path"].get<std::string>();
            if (!shared_face_det.empty()) break;
        }
    }

    // models
    out->sdk.models.clear();
    if (!j.contains("models") || !j["models"].is_array() || j["models"].empty())
        return fail("at least one model required");
    for (const auto& mo : j["models"]) {
        // face_rec/face_landmark 缺检测器路径：注入同任务 face_detection 模型路径（复合模型补齐）
        json mo_eff = mo;
        const bool is_face_rec = mo.is_object() && mo.contains("type") && mo["type"].is_string() &&
            normalize_model_type(mo["type"].get<std::string>()) == "face_rec";
        const bool is_face_landmark = mo.is_object() && mo.contains("type") &&
            mo["type"].is_string() &&
            normalize_model_type(mo["type"].get<std::string>()) == "face_landmark";
        if ((is_face_rec || is_face_landmark) && !shared_face_det.empty() &&
            !mo.contains("det_url") && !mo.contains("det_path")) {
            mo_eff["det_url"] = shared_face_det;
        }
        ModelConfig m;
        if (!parse_one_model(mo_eff, rect, &m, err)) return false;
        // sam/iseg：提示点模型条目优先，缺省用任务级 prompt_point（归一化 [x,y]）
        if (m.prompt_point.empty() && !task_prompt.empty() &&
            (m.type == "sam" || m.type == "iseg"))
            m.prompt_point = task_prompt;
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
    // 单事件人脸嵌入 Top-N 上限（云端顶层 int）：缺省 8（对齐 EventMeta::max_embeddings），
    // 夹紧到 [0, 64]（与云端 resolve_max_embeddings 口径一致）；<=0 表示禁用嵌入。
    out->max_embeddings = std::clamp(j.value("max_embeddings", 8), 0, 64);
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
    // 提示点（SAM_SEG；归一化 [x,y]）：支持 [x,y] 或 {"x":..,"y":..}，坐标钳制到 [0,1]
    if (mo.contains("prompt_point")) {
        const auto& pp = mo["prompt_point"];
        if (pp.is_array() && pp.size() >= 2 && pp[0].is_number() && pp[1].is_number())
            m.prompt_point = {std::clamp(pp[0].get<float>(), 0.f, 1.f),
                              std::clamp(pp[1].get<float>(), 0.f, 1.f)};
        else if (pp.is_object() && pp.contains("x") && pp.contains("y") &&
                 pp["x"].is_number() && pp["y"].is_number())
            m.prompt_point = {std::clamp(pp["x"].get<float>(), 0.f, 1.f),
                              std::clamp(pp["y"].get<float>(), 0.f, 1.f)};
    }
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
    // 多键别名解析：按顺序取第一个非空字符串键（解析规则与 resolve_path 一致）。
    // 云端复合模型的子模型键名可能因编排版本而不同，这里统一收口，避免静默丢模型。
    auto resolve_first = [&](std::initializer_list<const char*> keys, std::string* dst) -> bool {
        for (const char* k : keys) {
            if (mo.contains(k) && mo[k].is_string() && !mo[k].get<std::string>().empty())
                return resolve_path(k, dst);
        }
        return true;
    };
    if (m.type == "barcode") {
        // 条码/二维码：走原生 ZXing 解码，无需 DNN 权重。刻意不解析/下载 url：
        // 目录 BARCODE 的 pipeline 为 [ocr]（云端可能下发 OCR 权重），本族忽略该权重。
    } else if (m.type == "sam") {
        // 交互式分割（FastSAM-s）：单模型，提示点由 prompt_point 提供（已在上方解析）
        if (!resolve_first({"url", "path", "model_url", "model_path"}, &m.path)) return false;
    } else if (m.type == "face_rec") {
        // 人脸识别复合模型：det（SCRFD，复用 face_detection）+ rec（人脸嵌入，112→512-d）
        // det 缺省由 from_json 从同任务 face_detection 条目补齐（目录 pipeline 为两条独立模型）
        if (!resolve_first({"det_url", "det_path", "detector_url", "detector_path"}, &m.path))
            return false;
        if (!resolve_first({"rec_url", "rec_path", "embedding_url", "embedding_path",
                            "face_rec_url", "face_rec_path"}, &m.rec_path))
            return false;
        // 未给显式嵌入键时，url|path 即嵌入模型（云侧 face_rec 角色的主模型）
        if (m.rec_path.empty()) {
            if (!resolve_first({"url", "path"}, &m.rec_path)) return false;
        }
    } else if (m.type == "face_landmark") {
        // 人脸关键点复合模型：det（SCRFD，逐脸检测）+ landmark（2d106det，逐脸 106 点）。
        // 口径与 face_rec 对齐：det_url→path（检测器），landmark_url|url|path→rec_path（关键点模型）。
        // 无检测器时不写 rec_path，path 保持关键点模型（旧「整帧」口径，向后兼容）。
        std::string det, lmk;
        if (!resolve_first({"det_url", "det_path", "detector_url", "detector_path"}, &det))
            return false;
        if (!resolve_first({"landmark_url", "landmark_path", "lmk_url", "lmk_path",
                            "rec_url", "rec_path"}, &lmk))
            return false;
        if (lmk.empty()) {
            if (!resolve_first({"url", "path"}, &lmk)) return false;
        }
        if (!det.empty()) {
            // 复合口径：缺关键点模型必须显式失败（不把检测器当关键点模型加载）
            if (lmk.empty())
                return fail("model.landmark_url/url (face landmark) required for " + m.name);
            m.path = det;
            m.rec_path = lmk;
        } else {
            m.path = lmk;
        }
    } else if (m.type == "face_attr") {
        // 人脸性别/年龄复合模型：det（SCRFD）+ cls（性别）+ rec（年龄）
        if (!resolve_first({"det_url", "det_path", "url", "path"}, &m.path)) return false;
        if (!resolve_first({"cls_url", "cls_path", "gender_url", "gender_path"}, &m.cls_path))
            return false;
        if (!resolve_first({"rec_url", "rec_path", "age_url", "age_path"}, &m.rec_path))
            return false;
    } else if (m.type == "face_as") {
        // 人脸活体复合模型：det（SCRFD）+ cls（fas_first）+ rec（fas_second，可选）
        if (!resolve_first({"det_url", "det_path", "url", "path"}, &m.path)) return false;
        if (!resolve_first({"cls_url", "cls_path", "first_url", "first_path",
                            "fas_first_url", "fas_first_path"}, &m.cls_path))
            return false;
        if (!resolve_first({"rec_url", "rec_path", "second_url", "second_path",
                            "fas_second_url", "fas_second_path"}, &m.rec_path))
            return false;
    } else if (m.type == "ocr") {
        // OCR pipeline：det_url→path、cls_url→cls_path、rec_url→rec_path、dict_url→dict_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("cls_url", &m.cls_path)) return false;
        if (!resolve_path("rec_url", &m.rec_path)) return false;
        if (!resolve_path("dict_url", &m.dict_path)) return false;
    } else if (m.type == "lpr") {
        // LPR 车牌 pipeline：det_url→path、rec_url→rec_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("rec_url", &m.rec_path)) return false;
    } else if (m.type == "doc") {
        // 文档/表格：与 OCR 同口径（det 主模型 + cls 方向分类 + rec 识别 + 字典），
        // 识别文本写入 objects[].text（云侧 text_match/ocr_label 叶子可读）
        if (!resolve_first({"det_url", "det_path", "url", "path"}, &m.path)) return false;
        if (!resolve_first({"cls_url", "cls_path"}, &m.cls_path)) return false;
        if (!resolve_first({"rec_url", "rec_path"}, &m.rec_path)) return false;
        if (!resolve_first({"dict_url", "dict_path"}, &m.dict_path)) return false;
    } else {
        // pedestrian_attribute pipeline：det_url→path、cls_url→rec_path
        if (!resolve_path("det_url", &m.path)) return false;
        if (!resolve_path("cls_url", &m.rec_path)) return false;
    }
    // 单模型：path 优先；无 path 时本地/远端 url 均支持（远端经 ModelFetcher）
    // barcode 无权重：跳过路径解析（避免无谓下载），也不要求 path
    // face_rec 的 url|path 已解析为嵌入模型（rec_path），此处不得再当作检测器 path
    if (m.path.empty() && m.type != "barcode" && m.type != "face_rec") {
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
    if (m.path.empty() && m.type != "barcode")
        return fail("model.path/url required for " + m.name);
    // face_rec 还需嵌入模型（det 由 det_url 或同任务 face_detection 条目提供）
    if (m.type == "face_rec" && m.rec_path.empty())
        return fail("model.rec_url/url (face embedding) required for " + m.name);
    *out = std::move(m);
    return true;
}

bool ConfigAdapter::parse_model(const json& j, ModelConfig* out, std::string* err,
                                const std::vector<float>& roi_norm) const {
    return parse_one_model(j, roi_norm, out, err);
}

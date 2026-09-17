#include "event_bus.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>

namespace {

/// UTC ISO8601 毫秒（...Z）
std::string utc_iso8601_ms() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t secs = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(ms.count()));
    return std::string(buf);
}

/// float32 → IEEE-754 half（就近偶数舍入，含次正规/Inf/NaN）。
uint16_t f32_to_f16(float value) {
    uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t abs = x & 0x7FFFFFFFu;
    if (abs >= 0x7F800000u) {  // Inf / NaN
        return static_cast<uint16_t>(sign | (abs > 0x7F800000u ? 0x7E00u : 0x7C00u));
    }
    if (abs >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // 溢出 → Inf
    if (abs < 0x38800000u) {  // 次正规（< 2^-14）或零
        if (abs < 0x33000000u) return static_cast<uint16_t>(sign);  // < 2^-25 → ±0
        const uint32_t e = abs >> 23;                 // [102, 112]
        const uint32_t m = (abs & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - e;              // [14, 24]
        uint32_t half = m >> shift;
        const uint32_t round = (m >> (shift - 1)) & 1u;
        const uint32_t sticky = m & ((1u << (shift - 1)) - 1u);
        if (round && (sticky || (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    const uint32_t e = (abs >> 23) - 112u;            // half 指数 [1, 30]
    const uint32_t m = abs & 0x7FFFFFu;
    uint16_t half = static_cast<uint16_t>(sign | (e << 10) | (m >> 13));
    const uint32_t round = (m >> 12) & 1u;
    const uint32_t sticky = m & 0xFFFu;
    if (round && (sticky || (half & 1u))) ++half;
    return half;
}

/// IEEE-754 half → float32。
float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // 次正规 → 规格化
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float f = 0.f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/// 嵌入上限：仅保留置信度最高的 max_n 个对象的嵌入（分高者优先；<=0 全部移除）。
void apply_embedding_cap(std::vector<EventDetection>& dets, int max_n) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!dets[i].embedding.empty()) idx.push_back(i);
    }
    if (max_n <= 0) {
        for (size_t i : idx) dets[i].embedding.clear();
        return;
    }
    if (idx.size() <= static_cast<size_t>(max_n)) return;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return dets[a].confidence > dets[b].confidence;
    });
    for (size_t k = static_cast<size_t>(max_n); k < idx.size(); ++k) {
        dets[idx[k]].embedding.clear();
    }
}

}  // namespace

std::string encode_embedding_f16_b64(const std::vector<float>& vec) {
    if (vec.empty()) return "";
    std::vector<uint8_t> bytes;
    bytes.reserve(vec.size() * 2);
    for (float v : vec) {
        const uint16_t h = f32_to_f16(v);
        bytes.push_back(static_cast<uint8_t>(h & 0xFFu));
        bytes.push_back(static_cast<uint8_t>((h >> 8) & 0xFFu));
    }
    return aistation::base64_encode_bytes(bytes);
}

std::vector<float> decode_embedding_f16_b64(const std::string& text) {
    std::vector<float> out;
    const std::vector<uint8_t> bytes = aistation::base64_decode_bytes(text);
    if (bytes.empty() || bytes.size() % 2 != 0) return out;
    out.reserve(bytes.size() / 2);
    for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
        const uint16_t h = static_cast<uint16_t>(bytes[i]) |
                           static_cast<uint16_t>(static_cast<uint16_t>(bytes[i + 1]) << 8);
        out.push_back(f16_to_f32(h));
    }
    return out;
}


nlohmann::json DetectionEvent::to_json() const {
    nlohmann::json j;
    j["event_id"] = event_id;
    j["edge_code"] = edge_code;
    j["camera_id"] = camera_id;
    j["task_id"] = task_id;
    j["algorithm_type"] = algorithm_type;
    j["ts"] = ts;
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& d : detections) {
        nlohmann::json dj;
        dj["label"] = d.label;
        dj["label_id"] = d.label_id;
        dj["confidence"] = d.confidence;
        dj["bbox"] = {{"x", d.x}, {"y", d.y}, {"width", d.w}, {"height", d.h}};
        arr.push_back(std::move(dj));
    }
    j["detections"] = std::move(arr);
    // 事件 v2：objects（含属性）；detections 保留兼容
    nlohmann::json objects = nlohmann::json::array();
    for (const auto& d : detections) {
        nlohmann::json dj;
        dj["label"] = d.label;
        dj["label_id"] = d.label_id;
        dj["confidence"] = d.confidence;
        dj["bbox"] = {{"x", d.x}, {"y", d.y}, {"width", d.w}, {"height", d.h}};
        // 跟踪 ID：仅已分配（>=0）才写入
        if (d.track_id >= 0) dj["track_id"] = d.track_id;
        if (!d.attributes.empty()) {
            nlohmann::json attrs = nlohmann::json::object();
            for (const auto& [k, v] : d.attributes) attrs[k] = v;
            dj["attributes"] = std::move(attrs);
        }
        // OCR：文字非空才写入，避免污染纯检测事件
        if (!d.text.empty()) {
            dj["text"] = d.text;
            dj["text_score"] = d.text_score;
        }
        // 关键点（pose）：非空才写入，形态 [[x, y, score], ...]（归一化）；旧事件无此字段
        if (!d.keypoints.empty()) {
            nlohmann::json kps = nlohmann::json::array();
            for (const auto& k : d.keypoints) kps.push_back(nlohmann::json::array({k[0], k[1], k[2]}));
            dj["keypoints"] = std::move(kps);
        }
        // 深度（depth 模型）：仅在模型激活并产出有效距离时写入（增量字段，旧事件无此键）
        if (d.has_depth && std::isfinite(d.depth)) dj["depth"] = d.depth;
        // 人脸嵌入（face_rec）：紧凑编码 base64(float16) + 编码标识（增量字段，旧事件无此键）。
        // 原始 float 数组每脸约 10 KB（512-d，实测 9,940 B），24 张脸可达 ~235 KB，边缘上行不可接受；
        // f16b64 每脸 1,368 B，配合 Top-N（默认 8）把 24 脸事件的实测体积压到 19,925 B（~19.5 KB）。
        if (!d.embedding.empty()) {
            dj["embedding"] = encode_embedding_f16_b64(d.embedding);
            dj["embedding_encoding"] = "f16b64";
        }
        objects.push_back(std::move(dj));
    }
    j["objects"] = std::move(objects);
    j["latency_ms"] = latency_ms;
    if (!snapshot_data.empty()) {
        j["snapshot"] = {{"data", snapshot_data}, {"width", snapshot_width}, {"height", snapshot_height}};
    } else if (!snapshot_ref.empty()) {
        j["snapshot"] = {{"ref", snapshot_ref}};
    }
    j["schema_version"] = schema_version;
    // 心跳标记仅在为真时写入，普通检测事件契约保持不变
    if (heartbeat) j["heartbeat"] = true;
    return j;
}

DetectionEvent DetectionEvent::from_json(const nlohmann::json& j) {
    DetectionEvent e;
    e.event_id = j.value("event_id", "");
    e.edge_code = j.value("edge_code", "");
    e.camera_id = j.value("camera_id", 0);
    e.task_id = j.value("task_id", int64_t{0});
    e.algorithm_type = j.value("algorithm_type", "");
    e.ts = j.value("ts", "");
    e.latency_ms = j.value("latency_ms", 0.0);
    e.schema_version = j.value("schema_version", 1);
    e.heartbeat = j.value("heartbeat", false);
    if (j.contains("snapshot") && j["snapshot"].is_object()) {
        e.snapshot_ref = j["snapshot"].value("ref", "");
        e.snapshot_data = j["snapshot"].value("data", "");
        e.snapshot_width = j["snapshot"].value("width", 0);
        e.snapshot_height = j["snapshot"].value("height", 0);
    }
    if (j.contains("detections") && j["detections"].is_array()) {
        for (const auto& d : j["detections"]) {
            EventDetection ed;
            ed.label = d.value("label", "");
            ed.label_id = d.value("label_id", 0);
            ed.confidence = d.value("confidence", 0.f);
            if (d.contains("bbox")) {
                ed.x = d["bbox"].value("x", 0.f); ed.y = d["bbox"].value("y", 0.f);
                ed.w = d["bbox"].value("width", 0.f); ed.h = d["bbox"].value("height", 0.f);
            }
            e.detections.push_back(std::move(ed));
        }
    }
    // 事件 v2：detections[] 不携带 attributes/text，从对齐的 objects[] 按索引恢复
    if (j.contains("objects") && j["objects"].is_array()) {
        const auto& objs = j["objects"];
        for (size_t i = 0; i < e.detections.size() && i < objs.size(); ++i) {
            if (objs[i].contains("attributes") && objs[i]["attributes"].is_object()) {
                for (auto it = objs[i]["attributes"].begin(); it != objs[i]["attributes"].end(); ++it) {
                    if (it.value().is_number()) e.detections[i].attributes[it.key()] = it.value().get<float>();
                }
            }
            // OCR 文字与置信度
            e.detections[i].text = objs[i].value("text", "");
            e.detections[i].text_score = objs[i].value("text_score", 0.f);
            // 跟踪 ID（缺失视为未跟踪）
            e.detections[i].track_id = objs[i].value("track_id", -1);
            // 关键点（pose，归一化 [x, y, score]；缺失/形状异常跳过）
            if (objs[i].contains("keypoints") && objs[i]["keypoints"].is_array()) {
                auto& out = e.detections[i].keypoints;
                for (const auto& k : objs[i]["keypoints"]) {
                    if (k.is_array() && k.size() >= 3 && k[0].is_number() && k[1].is_number() &&
                        k[2].is_number())
                        out.push_back({k[0].get<float>(), k[1].get<float>(), k[2].get<float>()});
                }
            }
            // 深度（depth 模型，米；仅数值才恢复）
            if (objs[i].contains("depth") && objs[i]["depth"].is_number()) {
                e.detections[i].depth = objs[i]["depth"].get<float>();
                e.detections[i].has_depth = true;
            }
            // 人脸嵌入（face_rec）：
            // - f16b64：base64 字符串 + embedding_encoding 标识 → 解码回 float 向量；
            // - 原始 float 数组（无编码标识）→ 向后兼容，仅数值项才恢复，形状异常跳过。
            if (objs[i].contains("embedding")) {
                const auto& emb = objs[i]["embedding"];
                if (emb.is_string()) {
                    if (objs[i].value("embedding_encoding", "") == "f16b64") {
                        e.detections[i].embedding =
                            decode_embedding_f16_b64(emb.get<std::string>());
                    }
                } else if (emb.is_array()) {
                    auto& out = e.detections[i].embedding;
                    for (const auto& v : emb) {
                        if (v.is_number()) out.push_back(v.get<float>());
                    }
                }
            }
        }
    }
    return e;
}

std::string EventBus::make_uuid_v4() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t hi = dist(rng);
    uint64_t lo = dist(rng);
    // 版本 4：hi 的第 3 组高半字节置 0100
    hi = (hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;
    // 变体 10xx：lo 最高两位
    lo = (lo & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;

    char buf[40];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                  static_cast<unsigned>(hi >> 32),
                  static_cast<unsigned>((hi >> 16) & 0xFFFFu),
                  static_cast<unsigned>(hi & 0xFFFFu),
                  static_cast<unsigned>((lo >> 48) & 0xFFFFu),
                  static_cast<unsigned long long>(lo & 0x0000FFFFFFFFFFFFull));
    return std::string(buf);
}

void EventBus::register_task(const std::string& task_id, const EventMeta& meta) {
    std::lock_guard<std::mutex> lock(mtx_);
    TaskState st;
    st.meta = meta;
    tasks_[task_id] = std::move(st);
}

void EventBus::unregister_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.erase(task_id);
}

void EventBus::set_sink(EventSink sink) {
    std::lock_guard<std::mutex> lock(mtx_);
    sink_ = std::move(sink);
}

void EventBus::on_detections(const std::string& task_id, const std::vector<DetectionBox>& boxes,
                             int frame_w, int frame_h, double latency_ms) {
    if (frame_w <= 0 || frame_h <= 0 || boxes.empty()) return;

    DetectionEvent event;
    EventSink sink;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = tasks_.find(task_id);
        if (it == tasks_.end()) return;
        TaskState& st = it->second;
        if (!aistation::schedule_active_now(st.meta.schedule)) return;
        const auto now = std::chrono::steady_clock::now();

        std::vector<EventDetection> kept;
        std::vector<std::string> labels;
        kept.reserve(boxes.size());
        labels.reserve(boxes.size());
        for (const auto& b : boxes) {
            if (st.meta.alarm_interval_sec > 0) {
                auto lt = st.last_emit.find(b.label_name);
                if (lt != st.last_emit.end()) {
                    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                             now - lt->second).count();
                    if (elapsed < st.meta.alarm_interval_sec) continue;
                }
            }
            EventDetection ed;
            ed.label = b.label_name;
            ed.label_id = b.label_id;
            ed.confidence = b.score;
            ed.track_id = b.track_id;                 // 跟踪 ID 透传（-1 表示未跟踪）
            ed.x = b.x / static_cast<float>(frame_w);
            ed.y = b.y / static_cast<float>(frame_h);
            ed.w = b.w / static_cast<float>(frame_w);
            ed.h = b.h / static_cast<float>(frame_h);
            // 属性透传：DetectionBox 用 vector<pair>，事件用 map
            for (const auto& [k, v] : b.attributes) ed.attributes[k] = v;
            // OCR 文字透传
            ed.text = b.text;
            ed.text_score = b.text_score;
            // 关键点（pose）透传并归一化：像素 → [0,1]（分数原样；越界钳制保证契约 0..1）
            if (!b.keypoints.empty()) {
                ed.keypoints.reserve(b.keypoints.size());
                for (const auto& k : b.keypoints) {
                    ed.keypoints.push_back({
                        std::clamp(k[0] / static_cast<float>(frame_w), 0.f, 1.f),
                        std::clamp(k[1] / static_cast<float>(frame_h), 0.f, 1.f),
                        k[2],
                    });
                }
            }
            // 深度（depth 模型）透传：仅有效值才标记（非有限值丢弃，避免污染事件契约）
            if (b.has_depth && std::isfinite(b.depth)) {
                ed.has_depth = true;
                ed.depth = b.depth;
            }
            // 人脸嵌入（face_rec）透传：已 L2 归一化，非空才携带（增量字段，其它家族无此键）
            if (!b.embedding.empty()) ed.embedding = b.embedding;
            if (st.meta.roi_polygon.size() >= 3 &&
                !aistation::point_in_polygon(ed.x + ed.w * 0.5f, ed.y + ed.h * 0.5f, st.meta.roi_polygon)) {
                continue;
            }
            kept.push_back(std::move(ed));
            labels.push_back(b.label_name);
        }
        // 嵌入上限：仅置信度 Top-N 携带，约束最坏上行体积（<=0 表示禁用嵌入）
        apply_embedding_cap(kept, st.meta.max_embeddings);
        if (kept.empty()) return;

        event.event_id = make_uuid_v4();
        event.edge_code = st.meta.edge_code;
        event.camera_id = st.meta.camera_id;
        event.task_id = st.meta.task_id_num;
        event.algorithm_type = st.meta.algorithm_type;
        event.ts = utc_iso8601_ms();
        event.detections = std::move(kept);
        event.latency_ms = latency_ms;
        event.schema_version = 2;
        for (const auto& l : labels) st.last_emit[l] = now;
        sink = sink_;
    }
    if (sink) sink(event);
}

void EventBus::on_heartbeat(const std::string& task_id) {
    DetectionEvent event;
    EventSink sink;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = tasks_.find(task_id);
        if (it == tasks_.end()) return;
        TaskState& st = it->second;
        if (!aistation::schedule_active_now(st.meta.schedule)) return;

        event.event_id = make_uuid_v4();
        event.edge_code = st.meta.edge_code;
        event.camera_id = st.meta.camera_id;
        event.task_id = st.meta.task_id_num;
        event.algorithm_type = st.meta.algorithm_type;
        event.ts = utc_iso8601_ms();
        event.schema_version = 2;
        event.heartbeat = true;
        sink = sink_;
    }
    if (sink) sink(event);
}

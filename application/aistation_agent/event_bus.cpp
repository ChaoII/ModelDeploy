#include "event_bus.hpp"

#include <cstdio>
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

}  // namespace

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
    j["latency_ms"] = latency_ms;
    if (!snapshot_ref.empty()) j["snapshot"] = {{"ref", snapshot_ref}};
    j["schema_version"] = schema_version;
    return j;
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
            ed.x = b.x / static_cast<float>(frame_w);
            ed.y = b.y / static_cast<float>(frame_h);
            ed.w = b.w / static_cast<float>(frame_w);
            ed.h = b.h / static_cast<float>(frame_h);
            kept.push_back(std::move(ed));
            labels.push_back(b.label_name);
        }
        if (kept.empty()) return;

        event.event_id = make_uuid_v4();
        event.edge_code = st.meta.edge_code;
        event.camera_id = st.meta.camera_id;
        event.task_id = st.meta.task_id_num;
        event.algorithm_type = st.meta.algorithm_type;
        event.ts = utc_iso8601_ms();
        event.detections = std::move(kept);
        event.latency_ms = latency_ms;
        event.schema_version = 1;
        for (const auto& l : labels) st.last_emit[l] = now;
        sink = sink_;
    }
    if (sink) sink(event);
}

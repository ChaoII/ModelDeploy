#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "inference_engine.hpp"

/// 单个检测框（归一化坐标）
struct EventDetection {
    std::string label;
    int label_id = 0;
    float confidence = 0.f;
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f;   // 归一化
};

/// 检测事件（Agent → 云端，见设计 §7.2）
struct DetectionEvent {
    std::string event_id;
    std::string edge_code;
    int camera_id = 0;
    int64_t task_id = 0;
    std::string algorithm_type;
    std::string ts;                              // UTC ISO8601 毫秒 Z
    std::vector<EventDetection> detections;
    double latency_ms = 0.0;
    std::string snapshot_ref;                    // 可空
    int schema_version = 1;

    nlohmann::json to_json() const;
};

/// 每任务事件元数据
struct EventMeta {
    std::string edge_code;
    int camera_id = 0;
    int64_t task_id_num = 0;
    std::string algorithm_type;                  // AIStation 提供，原样带回
    int alarm_interval_sec = 0;                  // 0 = 不节流
};

using EventSink = std::function<void(const DetectionEvent&)>;

/// 每任务检测回调注册与分发（label 维度节流 + 事件装配）
class EventBus {
public:
    EventBus() = default;

    void register_task(const std::string& task_id, const EventMeta& meta);
    void unregister_task(const std::string& task_id);
    void set_sink(EventSink sink);               // 指向发布器
    void on_detections(const std::string& task_id, const std::vector<DetectionBox>& boxes,
                       int frame_w, int frame_h, double latency_ms);

    /// RFC4122 v4 UUID（36 字符，含连字符）
    static std::string make_uuid_v4();

private:
    struct TaskState {
        EventMeta meta;
        std::map<std::string, std::chrono::steady_clock::time_point> last_emit;   // label -> 时间
    };

    std::mutex mtx_;
    std::map<std::string, TaskState> tasks_;
    EventSink sink_;
};

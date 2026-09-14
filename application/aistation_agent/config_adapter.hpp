#pragma once
#include <array>
#include <string>
#include <vector>
#include "agent_util.hpp"
#include "config.hpp"
#include "model_fetcher.hpp"
#include "nlohmann/json.hpp"

struct CameraMeta {
    int id = 0;
    std::string name;
    std::string url;
    std::string transport = "tcp";
};

/// 事件内联快照配置（Agent 采集最新帧 JPEG → base64 随事件上报）
struct SnapshotConfig {
    bool enabled = false;
    bool inline_snapshot = false;
    int quality = 75;
    int max_width = 640;
};

struct EventPublishConfig {
    std::string transport = "http";        // mqtt | http
    // mqtt
    std::string mqtt_broker;
    std::string mqtt_topic;
    std::string mqtt_client_id;
    std::string mqtt_username;
    std::string mqtt_password;
    int mqtt_qos = 1;
    // http
    std::string http_url;
    std::string http_token;
    // buffer
    std::string buffer_dir = "./events_buffer";
    int buffer_max_mb = 512;
};

struct AdaptedTask {
    TaskConfig sdk;                        // SDK 任务配置
    CameraMeta camera;
    std::string algorithm_type;            // AIStation 提供，原样带回
    int alarm_interval_sec = 0;
    EventPublishConfig events;
    std::string tenant = "default";
    aistation::Schedule schedule;                              // 布控时段（空=全天）
    std::vector<std::array<float, 2>> roi_polygon;             // 归一化多边形（<3 顶点=不过滤）
    SnapshotConfig snapshot;                                   // 事件内联快照
    TaskConfig::TrackingConfig tracking;                       // 跟踪配置（顶层 tracking）
};

/// AIStation TaskConfig(JSON) → SDK TaskConfig + Agent 元数据
class ConfigAdapter {
public:
    explicit ConfigAdapter(ModelFetcher* fetcher = nullptr) : fetcher_(fetcher) {}
    bool from_json(const nlohmann::json& j, AdaptedTask* out, std::string* err) const;

    /// 归一化多边形 → 外接矩形 [x,y,w,h]（归一化）；非法返回空
    static std::vector<float> polygon_to_norm_rect(const nlohmann::json& roi);

private:
    ModelFetcher* fetcher_ = nullptr;
};

/// 模型 type 归一化：det→detection, cls→classification, face→face_detection（未知原样）
std::string normalize_model_type(const std::string& type);

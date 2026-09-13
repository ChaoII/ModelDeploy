#pragma once
#include <string>
#include <vector>
#include "config.hpp"
#include "nlohmann/json.hpp"

struct CameraMeta {
    int id = 0;
    std::string name;
    std::string url;
    std::string transport = "tcp";
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
};

/// AIStation TaskConfig(JSON) → SDK TaskConfig + Agent 元数据
class ConfigAdapter {
public:
    explicit ConfigAdapter(std::string model_cache_dir = "") : model_cache_dir_(std::move(model_cache_dir)) {}
    bool from_json(const nlohmann::json& j, AdaptedTask* out, std::string* err) const;

    /// 归一化多边形 → 外接矩形 [x,y,w,h]（归一化）；非法返回空
    static std::vector<float> polygon_to_norm_rect(const nlohmann::json& roi);

private:
    std::string model_cache_dir_;
};

/// 模型 type 归一化：det→detection, cls→classification, face→face_detection（未知原样）
std::string normalize_model_type(const std::string& type);

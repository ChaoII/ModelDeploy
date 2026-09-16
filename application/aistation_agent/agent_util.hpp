#pragma once
#include <array>
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace aistation {

struct ScheduleSlot {
    int day = 0;          // 0=周一 .. 6=周日
    int start_hour = 0;   // [start, end) 整点
    int end_hour = 24;
};

struct Schedule {
    std::vector<ScheduleSlot> slots;  // 空 = 全天有效
};

Schedule parse_schedule(const nlohmann::json& j);
bool schedule_active(const Schedule& s, int iso_day, int hour);
/// 按 UTC 时刻判定时段是否生效（与云端 gating 一致）；utc_t 为 Unix 秒。
bool schedule_active_at(const Schedule& s, std::time_t utc_t);
bool schedule_active_now(const Schedule& s);

/// 心跳是否到期：interval_sec<=0 禁用；last 为默认值(epoch 0) 视为立即到期。
bool heartbeat_due(std::chrono::steady_clock::time_point last, int interval_sec,
                   std::chrono::steady_clock::time_point now);

bool point_in_polygon(float px, float py, const std::vector<std::array<float, 2>>& poly);
std::vector<std::array<float, 2>> parse_roi_polygon(const nlohmann::json& roi);

std::string normalize_broker_scheme(const std::string& url);
std::string base64_encode_bytes(const std::vector<uint8_t>& data);

/// 计算某任务实际使用的 MQTT client_id（防御云端下发"每边缘唯一"而非"每任务唯一"的 id）。
/// Agent 为**每个任务**创建独立的 MQTT 连接，故 client_id 必须在同一 edge 内唯一：MQTT 规范
/// 要求 broker 在收到重复 client_id 的连接时断开旧连接，若所有任务共用同一 id，会造成任务间
/// 相互顶下线、事件丢失。这里在云端下发值（或默认 aistation-agent-<edge_code>）后统一追加
/// 任务 id，保证每任务唯一且跨重连稳定。configured 为空时使用 edge_code 派生默认前缀。
/// 若拼接结果超过 23 字节（MQTT 3.1.1 [MQTT-3.1.3-5] 强制 broker 支持的上限），则确定性压缩为
/// `aistation-agent-<7 位十六进制 FNV-1a>`（恒 23 字节），同输入稳定、异输入几乎不撞。
std::string mqtt_client_id_for(const std::string& configured, const std::string& edge_code,
                               const std::string& task_id);

}  // namespace aistation

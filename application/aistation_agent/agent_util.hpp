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
bool schedule_active_now(const Schedule& s);

/// 心跳是否到期：interval_sec<=0 禁用；last 为默认值(epoch 0) 视为立即到期。
bool heartbeat_due(std::chrono::steady_clock::time_point last, int interval_sec,
                   std::chrono::steady_clock::time_point now);

bool point_in_polygon(float px, float py, const std::vector<std::array<float, 2>>& poly);
std::vector<std::array<float, 2>> parse_roi_polygon(const nlohmann::json& roi);

std::string normalize_broker_scheme(const std::string& url);
std::string base64_encode_bytes(const std::vector<uint8_t>& data);

}  // namespace aistation

#include "agent_util.hpp"

#include <algorithm>
#include <ctime>

#include "httplib.h"

namespace aistation {

Schedule parse_schedule(const nlohmann::json& j) {
    Schedule s;
    const auto it = j.find("slots");
    if (it == j.end() || !it->is_array()) return s;  // 缺失/非法 = 全天
    for (const auto& e : *it) {
        if (!e.is_object()) continue;
        ScheduleSlot sl;
        sl.day = e.value("day", -1);
        sl.start_hour = std::clamp(e.value("start", 0), 0, 24);
        sl.end_hour = std::clamp(e.value("end", 24), 0, 24);
        if (sl.day < 0 || sl.day > 6) continue;
        if (sl.end_hour <= sl.start_hour) continue;
        s.slots.push_back(sl);
    }
    return s;
}

bool schedule_active(const Schedule& s, int iso_day, int hour) {
    if (s.slots.empty()) return true;
    for (const auto& sl : s.slots) {
        if (sl.day == iso_day && hour >= sl.start_hour && hour < sl.end_hour) return true;
    }
    return false;
}

bool schedule_active_now(const Schedule& s) {
    if (s.slots.empty()) return true;
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    const int iso_day = (tm.tm_wday + 6) % 7;  // tm_wday: 0=周日 -> 周一=0
    return schedule_active(s, iso_day, tm.tm_hour);
}

bool point_in_polygon(float px, float py, const std::vector<std::array<float, 2>>& poly) {
    const size_t n = poly.size();
    if (n < 3) return true;  // 退化多边形 = 不过滤
    bool inside = false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const float xi = poly[i][0], yi = poly[i][1];
        const float xj = poly[j][0], yj = poly[j][1];
        const bool intersects = ((yi > py) != (yj > py)) &&
                                (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersects) inside = !inside;
    }
    return inside;
}

std::vector<std::array<float, 2>> parse_roi_polygon(const nlohmann::json& roi) {
    std::vector<std::array<float, 2>> out;
    if (!roi.is_array()) return out;
    for (const auto& p : roi) {
        if (!p.is_array() || p.size() < 2 || !p[0].is_number() || !p[1].is_number()) continue;
        out.push_back({std::clamp(p[0].get<float>(), 0.f, 1.f),
                       std::clamp(p[1].get<float>(), 0.f, 1.f)});
    }
    return out;
}

std::string normalize_broker_scheme(const std::string& url) {
    if (url.rfind("mqtts://", 0) == 0) return "ssl://" + url.substr(8);
    if (url.rfind("mqtt://", 0) == 0) return "tcp://" + url.substr(7);
    if (url.find("://") == std::string::npos && !url.empty()) return "tcp://" + url;
    return url;
}

std::string base64_encode_bytes(const std::vector<uint8_t>& data) {
    if (data.empty()) return "";
    return httplib::detail::base64_encode(
        std::string(reinterpret_cast<const char*>(data.data()), data.size()));
}

}  // namespace aistation

#include "agent_util.hpp"

#include <algorithm>
#include <cstdint>
#include <ctime>

#include "httplib.h"

namespace {

/// 时间点（UTC）拆分为 ISO 星期与小时：周一=0 .. 周日=6，hour ∈ [0,23]。
void utc_day_hour(std::time_t t, int& iso_day, int& hour) {
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    iso_day = (tm.tm_wday + 6) % 7;  // tm_wday: 0=周日 -> 周一=0
    hour = tm.tm_hour;
}

/// MQTT 3.1.1 [MQTT-3.1.3-5]：broker 必须支持 1..23 字节的 client_id。
/// 云端 edge_code 为 String(64)，与任务 id 拼接后可能远超 23 字节（如 `aistation-agent-edge-01-101`
/// 已 27 字节），故超长时压缩为固定长度，避免依赖 broker 的非强制扩展行为。
constexpr std::size_t kMqttClientIdMax = 23;
/// 压缩形式长度：`aistation-agent-`（16）+ 7 位十六进制哈希 = 23。
constexpr std::size_t kMqttClientIdHashChars = 7;

/// FNV-1a 64 位：跨平台/跨进程确定性的稳定哈希（不用 std::hash，后者实现相关、不保证稳定）。
std::uint64_t fnv1a64(const std::string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

/// 超长 client_id 的确定性压缩：固定前缀 + 7 位十六进制哈希（总长恒为 kMqttClientIdMax）。
std::string shorten_mqtt_client_id(const std::string& full) {
    static const char kHex[] = "0123456789abcdef";
    const std::uint64_t v = fnv1a64(full);
    std::string out = "aistation-agent-";
    for (int i = static_cast<int>(kMqttClientIdHashChars) - 1; i >= 0; --i) {
        out.push_back(kHex[(v >> (4 * i)) & 0xFULL]);
    }
    return out;
}

}  // namespace

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

bool schedule_active_at(const Schedule& s, std::time_t utc_t) {
    if (s.slots.empty()) return true;
    int iso_day = 0;
    int hour = 0;
    utc_day_hour(utc_t, iso_day, hour);
    return schedule_active(s, iso_day, hour);
}

bool schedule_active_now(const Schedule& s) {
    // 与云端 gating 一致，统一按 UTC 判定（云端 inference/gating.py 用
    // datetime.fromtimestamp(now, tz=timezone.utc) 推导 day/hour）。若改用 localtime，
    // 非 UTC 主机的同一时段窗口会对应不同的墙钟时间，与云端判定不一致。
    return schedule_active_at(s, std::time(nullptr));
}

bool heartbeat_due(std::chrono::steady_clock::time_point last, int interval_sec,
                   std::chrono::steady_clock::time_point now) {
    if (interval_sec <= 0) return false;
    if (last.time_since_epoch().count() == 0) return true;
    return std::chrono::duration_cast<std::chrono::seconds>(now - last).count() >= interval_sec;
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

namespace {

/// 单个 base64 字符 → 6bit 值；非法字符返回 -1。
int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

}  // namespace

std::vector<uint8_t> base64_decode_bytes(const std::string& text) {
    std::vector<uint8_t> out;
    if (text.empty()) return out;
    out.reserve(text.size() / 4 * 3);
    uint32_t buffer = 0;
    int bits = 0;
    for (char c : text) {
        if (c == '=') break;  // 填充开始，忽略其后
        // 容忍换行/空白（不同实现可能插入）
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = base64_value(c);
        if (v < 0) return {};  // 非法字符 → 整体失败
        buffer = (buffer << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFFu));
        }
    }
    return out;
}

std::string mqtt_client_id_for(const std::string& configured, const std::string& edge_code,
                               const std::string& task_id) {
    // 云端（orchestrator）当前对所有任务下发同一个 `aistation-agent-<edge_code>`（每边缘唯一），
    // 但本 Agent 每任务一条 MQTT 连接，故必须在此追加任务 id 才不会相互顶下线（见头文件说明）。
    const std::string base =
        configured.empty() ? "aistation-agent-" + edge_code : configured;
    const std::string full = task_id.empty() ? base : base + "-" + task_id;
    // 超过 23 字节（MQTT 3.1.1 强制 broker 支持的上限）时确定性压缩：同输入稳定、异输入几乎不撞。
    if (full.size() <= kMqttClientIdMax) return full;
    return shorten_mqtt_client_id(full);
}

}  // namespace aistation

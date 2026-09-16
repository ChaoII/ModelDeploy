#include <catch2/catch_test_macros.hpp>
#include <array>
#include <chrono>
#include <vector>
#include "agent_util.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
using namespace aistation;

TEST_CASE("parse_schedule handles slots and empty", "[agent][util]") {
    Schedule empty = parse_schedule(json::object());
    REQUIRE(empty.slots.empty());

    Schedule s = parse_schedule(json::parse(R"({"slots":[{"day":2,"start":8,"end":18}]})"));
    REQUIRE(s.slots.size() == 1);
    REQUIRE(s.slots[0].day == 2);
    REQUIRE(s.slots[0].start_hour == 8);
    REQUIRE(s.slots[0].end_hour == 18);

    // 非法 slot（day 越界 / end<=start）被丢弃
    Schedule bad = parse_schedule(json::parse(R"({"slots":[{"day":9,"start":1,"end":2},{"day":1,"start":5,"end":5}]})"));
    REQUIRE(bad.slots.empty());
}

TEST_CASE("schedule_active window boundaries", "[agent][util]") {
    Schedule s;
    s.slots.push_back({0, 8, 18});  // 周一 08:00-18:00
    REQUIRE(schedule_active(s, 0, 8));        // 起始含
    REQUIRE(schedule_active(s, 0, 17));       // 末内
    REQUIRE_FALSE(schedule_active(s, 0, 18)); // 结束不含
    REQUIRE_FALSE(schedule_active(s, 1, 10)); // 非当日
    Schedule all;  // 空 slots
    REQUIRE(schedule_active(all, 3, 0));
}

TEST_CASE("schedule_active_at derives day/hour from UTC", "[agent][util]") {
    // 1700000000 = 2023-11-14T22:13:20Z（周二 22 点）。本机为 UTC+8 时，
    // localtime 会得到周三 06:13，故该用例可稳定区分 UTC 与本地时区判定。
    const std::time_t t = 1700000000;
    Schedule utc_hit;                        // 周二 22:00-23:00（UTC）
    utc_hit.slots.push_back({1, 22, 23});
    REQUIRE(schedule_active_at(utc_hit, t));

    Schedule local_hit;                      // 周三 06:00-07:00（本地时区，非 UTC）
    local_hit.slots.push_back({2, 6, 7});
    REQUIRE_FALSE(schedule_active_at(local_hit, t));

    Schedule all;                            // 空 slots = 全天
    REQUIRE(schedule_active_at(all, t));
}

TEST_CASE("schedule_active_now aligns with UTC", "[agent][util]") {
    // 用 gmtime 独立推导期望的 UTC 星期/小时，确保实现不依赖 localtime。
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    const int iso_day = (tm.tm_wday + 6) % 7;
    const int hour = tm.tm_hour;

    Schedule hit;
    hit.slots.push_back({iso_day, hour, hour + 1});
    REQUIRE(schedule_active_now(hit));

    Schedule miss;  // 三天后的同一小时必定不在窗口内
    miss.slots.push_back({(iso_day + 3) % 7, hour, hour + 1});
    REQUIRE_FALSE(schedule_active_now(miss));

    Schedule all;  // 空 slots = 全天
    REQUIRE(schedule_active_now(all));
}

TEST_CASE("point_in_polygon inside/outside/degenerate", "[agent][util]") {
    std::vector<std::array<float, 2>> poly = {{{0.1f,0.1f}},{{0.9f,0.1f}},{{0.9f,0.9f}},{{0.1f,0.9f}}};
    REQUIRE(point_in_polygon(0.5f, 0.5f, poly));
    REQUIRE_FALSE(point_in_polygon(0.95f, 0.5f, poly));
    std::vector<std::array<float, 2>> degenerate = {{{0.1f,0.1f}},{{0.9f,0.9f}}};
    REQUIRE(point_in_polygon(0.95f, 0.5f, degenerate));  // <3 顶点=不过滤
}

TEST_CASE("normalize_broker_scheme", "[agent][util]") {
    REQUIRE(normalize_broker_scheme("mqtt://h:1883") == "tcp://h:1883");
    REQUIRE(normalize_broker_scheme("mqtts://h:8883") == "ssl://h:8883");
    REQUIRE(normalize_broker_scheme("h:1883") == "tcp://h:1883");
    REQUIRE(normalize_broker_scheme("tcp://h:1883") == "tcp://h:1883");
    REQUIRE(normalize_broker_scheme("") == "");
}

TEST_CASE("base64_encode_bytes", "[agent][util]") {
    std::vector<uint8_t> data = {'M', 'a', 'n'};
    REQUIRE(base64_encode_bytes(data) == "TWFu");
}

TEST_CASE("heartbeat_due honors interval and disable", "[agent][heartbeat]") {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    // 首次（last 为默认构造）立即到期
    REQUIRE(aistation::heartbeat_due(clock::time_point{}, 5, t0) == true);
    // 未到间隔
    REQUIRE(aistation::heartbeat_due(t0, 5, t0 + std::chrono::seconds(4)) == false);
    // 刚好到间隔
    REQUIRE(aistation::heartbeat_due(t0, 5, t0 + std::chrono::seconds(5)) == true);
    // <=0 禁用
    REQUIRE(aistation::heartbeat_due(t0, 0, t0 + std::chrono::seconds(60)) == false);
    REQUIRE(aistation::heartbeat_due(t0, -1, t0 + std::chrono::seconds(60)) == false);
}

// MQTT client_id 唯一化：Agent 每任务一条 MQTT 连接，而云端对所有任务下发同一个
// "aistation-agent-<edge_code>"（每边缘唯一）。若原样使用，broker 会按 MQTT 规范在收到重复
// client_id 时断开旧连接 → 同一边缘的多任务相互顶下线、事件丢失。这里断言追加任务 id 后
// 每任务唯一、同任务稳定（跨重连/热更新不变）、云端未下发时默认前缀同样唯一。
TEST_CASE("mqtt_client_id_for makes per-task ids unique and stable", "[agent][util][mqtt]") {
    const std::string cloud = "aistation-agent-edge-01";
    const std::string a = aistation::mqtt_client_id_for(cloud, "edge-01", "101");
    const std::string b = aistation::mqtt_client_id_for(cloud, "edge-01", "102");
    REQUIRE(a == cloud + "-101");
    REQUIRE(a != b);                                                // 同边缘不同任务互不冲突
    REQUIRE(aistation::mqtt_client_id_for(cloud, "edge-01", "101") == a);  // 同任务稳定

    const std::string d1 = aistation::mqtt_client_id_for("", "edge-01", "101");
    const std::string d2 = aistation::mqtt_client_id_for("", "edge-01", "102");
    REQUIRE(d1 == "aistation-agent-edge-01-101");
    REQUIRE(d1 != d2);

    // 空任务 id 时退化为前缀本身（不应出现，但不产生非法空 id）
    REQUIRE(aistation::mqtt_client_id_for(cloud, "edge-01", "") == cloud);
}

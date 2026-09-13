#include <catch2/catch_test_macros.hpp>
#include <array>
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

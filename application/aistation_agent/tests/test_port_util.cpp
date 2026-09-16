#include <catch2/catch_test_macros.hpp>
#include <set>

#include "test_port_util.hpp"

// 回归测试：端口分配器必须做到"进程内唯一 + 分配时可绑定"。
// 原实现（各测试文件各自 `base + rng() % span`）无法保证同一测试内两处调用返回不同端口；
// 在 httplib SO_REUSEADDR（Windows 下先绑定者优先）下，这会让后启动的 Agent 控制面被
// 先启动的事件接收器遮蔽，表现为偶发、不可按需复现的 POST /tasks 失败。
TEST_CASE("test port allocator hands out distinct and bindable ports", "[agent][port]") {
    std::set<int> seen;
    for (int i = 0; i < 64; ++i) {
        const int p = aistation_test::test_free_port();
        REQUIRE(p >= aistation_test::kPortBase);
        REQUIRE(p < aistation_test::kPortBase + aistation_test::kPortSpan);
        REQUIRE(seen.insert(p).second);             // 进程内绝不重复（消除自碰撞）
        REQUIRE(aistation_test::port_bindable(p));  // 分配时端口确实可绑定
    }
}

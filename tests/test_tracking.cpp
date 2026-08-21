#include <catch2/catch_test_macros.hpp>
#include "vision/tracking/base_tracker.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tracking;

TEST_CASE("BaseTracker: empty input yields empty output", "[tracking]") {
    auto r = empty_update();
    REQUIRE(r.empty());
}

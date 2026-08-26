#include "vision/processors/cuda/cjk_font_util.h"
#include <catch2/catch_test_macros.hpp>
namespace mv = modeldeploy::vision;
TEST_CASE("cjk_font: lookup and utf8", "[cjkfont][core]") {
    uint32_t cp = 0;
    REQUIRE(mv::utf8_to_cp("A", &cp) == 1);
    REQUIRE(cp == 0x41);
    const char* zi = "\xE4\xB8\xAD";  // "中"
    REQUIRE(mv::utf8_to_cp(zi, &cp) == 3);
    REQUIRE(cp == 0x4E2D);
    REQUIRE(mv::cjk_lookup(0x41) >= 0);
    REQUIRE(mv::cjk_lookup(0x4E2D) >= 0);
}

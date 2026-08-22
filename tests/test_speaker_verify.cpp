#include <catch2/catch_test_macros.hpp>
#include "csrc/audio/speaker_gallery.h"

using namespace modeldeploy::audio;

TEST_CASE("SpeakerGallery enroll/match/remove", "[speaker]") {
    SpeakerGallery g;
    g.enroll("alice", std::vector<float>{1.0f, 0.0f, 0.0f});
    g.enroll("bob", std::vector<float>{0.0f, 1.0f, 0.0f});
    REQUIRE(g.size() == 2);
    auto top = g.match(std::vector<float>{0.99f, 0.01f, 0.0f}, 1);
    REQUIRE(top.size() == 1);
    REQUIRE(top[0].first == "alice");
    REQUIRE(top[0].second > 0.9f);
    g.remove("alice");
    REQUIRE(g.size() == 1);
    g.clear();
    REQUIRE(g.size() == 0);
}

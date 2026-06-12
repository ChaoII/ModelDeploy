#include <catch2/catch_test_macros.hpp>
#include "stream_decoder.hpp"

TEST_CASE("StreamDecoder construct and cleanup", "[decoder]") {
    DecoderConfig cfg;
    StreamDecoder dec(cfg);
    REQUIRE_FALSE(dec.is_running());
}

TEST_CASE("StreamDecoder start without open", "[decoder]") {
    DecoderConfig cfg;
    StreamDecoder dec(cfg);
    REQUIRE_FALSE(dec.start());
}

TEST_CASE("StreamDecoder stop when not running", "[decoder]") {
    DecoderConfig cfg;
    StreamDecoder dec(cfg);
    REQUIRE_NOTHROW(dec.stop());
}

TEST_CASE("StreamDecoder callback set/get", "[decoder]") {
    DecoderConfig cfg;
    StreamDecoder dec(cfg);
    int called = 0;
    dec.set_callback([&](const DecodedFrame&) { ++called; return true; });
    // Can't test actual callback without opening a stream
    REQUIRE_NOTHROW(dec.set_callback(nullptr));
}

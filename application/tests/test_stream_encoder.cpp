#include <catch2/catch_test_macros.hpp>
#include "stream_encoder.hpp"
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy::vision;

TEST_CASE("Encoder construct/destroy", "[encoder]") {
    StreamEncoder enc(25);
    REQUIRE_FALSE(enc.is_open());
}

TEST_CASE("Encoder open/close cycle", "[encoder]") {
    StreamEncoder enc(25);
    // Can't test actual RTSP push without a server
    // Just verify it doesn't crash on close when not opened
    REQUIRE_NOTHROW(enc.close());
}

TEST_CASE("Encoder double close", "[encoder]") {
    StreamEncoder enc(25);
    REQUIRE_NOTHROW(enc.close());
    REQUIRE_NOTHROW(enc.close());
}

TEST_CASE("Encoder async without open", "[encoder]") {
    StreamEncoder enc(25);
    REQUIRE_FALSE(enc.start_async());
    REQUIRE_NOTHROW(enc.stop_async());
}

TEST_CASE("Encoder encode without open", "[encoder]") {
    StreamEncoder enc(25);
    ImageData img(100, 100, MdImageType::PKG_BGR_U8);
    REQUIRE_FALSE(enc.encode(img));
}

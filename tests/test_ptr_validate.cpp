#include <catch2/catch_test_macros.hpp>
#include "core/device_validate.h"

using namespace modeldeploy;

TEST_CASE("validate_pointer CPU non-null -> Ok", "[core]") {
    int x = 0;
    DeviceValidateCode c = DeviceValidateCode::Invalid;
    REQUIRE(validate_pointer_device(&x, Device::CPU, 0, &c));
    REQUIRE(c == DeviceValidateCode::Ok);
}
TEST_CASE("validate_pointer CPU null -> Invalid", "[core]") {
    DeviceValidateCode c = DeviceValidateCode::Ok;
    REQUIRE_FALSE(validate_pointer_device(nullptr, Device::CPU, 0, &c));
    REQUIRE(c == DeviceValidateCode::Invalid);
}
TEST_CASE("validate_pointer TPU -> Unsupported", "[core]") {
    int x = 0;
    DeviceValidateCode c = DeviceValidateCode::Ok;
    REQUIRE_FALSE(validate_pointer_device(&x, Device::TPU, 0, &c));
    REQUIRE(c == DeviceValidateCode::Unsupported);
}
TEST_CASE("validate_pointer OPENCL -> Unsupported", "[core]") {
    int x = 0;
    DeviceValidateCode c = DeviceValidateCode::Ok;
    REQUIRE_FALSE(validate_pointer_device(&x, Device::OPENCL, 0, &c));
    REQUIRE(c == DeviceValidateCode::Unsupported);
}

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"

using namespace modeldeploy;

TEST_CASE("Tensor basic operations", "[core]") {
    Tensor t;
    REQUIRE(t.shape().size() == 1);
    REQUIRE(t.shape()[0] == 0);
    REQUIRE(t.byte_size() == 0);

    t.allocate({1, 3, 224, 224}, DataType::FP32);
    REQUIRE(t.shape().size() == 4);
    REQUIRE(t.shape()[0] == 1);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.shape()[2] == 224);
    REQUIRE(t.shape()[3] == 224);
    REQUIRE(t.byte_size() == 1 * 3 * 224 * 224 * 4);
    REQUIRE(t.data() != nullptr);
}

TEST_CASE("Tensor dtype float access", "[core]") {
    Tensor t;
    t.allocate({10}, DataType::FP32);
    REQUIRE(t.dtype() == DataType::FP32);
    REQUIRE(t.byte_size() == 10 * 4);

    auto* data = static_cast<float*>(t.data());
    REQUIRE(data != nullptr);
    for (int i = 0; i < 10; ++i) data[i] = static_cast<float>(i * 100);

    Tensor t2(t);
    auto* data2 = static_cast<float*>(t2.data());
    REQUIRE(data2 != nullptr);
    REQUIRE(data2[5] == 500.0f);
}

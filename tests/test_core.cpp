#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "utils/benchmark.h"

using namespace modeldeploy;

TEST_CASE("Tensor basic operations", "[core]") {
    Tensor t;
    REQUIRE(t.shape().empty());
    REQUIRE(t.nbytes() == 0);

    t.allocate({1, 3, 224, 224}, DataType::FP32);
    REQUIRE(t.shape().size() == 4);
    REQUIRE(t.shape()[0] == 1);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.shape()[2] == 224);
    REQUIRE(t.shape()[3] == 224);
    REQUIRE(t.nbytes() == 1 * 3 * 224 * 224 * 4);
    REQUIRE(t.data() != nullptr);
}

TEST_CASE("Tensor dtype conversions", "[core]") {
    Tensor t;
    t.allocate({10}, DataType::INT64);
    REQUIRE(t.dtype() == DataType::INT64);
    REQUIRE(t.nbytes() == 10 * 8);

    auto* data = t.data<int64_t>();
    REQUIRE(data != nullptr);
    for (int i = 0; i < 10; ++i) data[i] = i * 100;

    Tensor t2(t);
    auto* data2 = t2.data<int64_t>();
    REQUIRE(data2 != nullptr);
    REQUIRE(data2[5] == 500);
}

TEST_CASE("Timer basic", "[core]") {
    Timer timer;
    timer.start();
    volatile int sum = 0;
    for (int i = 0; i < 10000; ++i) sum += i;
    timer.stop();
    REQUIRE(timer.average_ms() >= 0);
}

TEST_CASE("TimerArray basic", "[core]") {
    TimerArray arr;
    arr.pre_timer.start();
    arr.pre_timer.stop();
    arr.infer_timer.start();
    arr.infer_timer.stop();
    arr.post_timer.start();
    arr.post_timer.stop();
    REQUIRE(arr.total_ms() >= 0);
}

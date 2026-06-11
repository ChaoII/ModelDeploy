#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>
#include <cstring>
#include "core/tensor.h"
#include "core/enum_variables.h"

using namespace modeldeploy;

// ============ Tensor construction ============

TEST_CASE("Tensor default constructor", "[core]") {
    Tensor t;
    REQUIRE(t.shape().size() == 1);
    REQUIRE(t.shape()[0] == 0);
    REQUIRE(t.size() == 0);
    REQUIRE(t.byte_size() == 0);
    REQUIRE(t.dtype() == DataType::FP32);
    REQUIRE(t.device() == Device::CPU);
    REQUIRE(t.data() == nullptr);
    REQUIRE(t.get_name().empty());
    REQUIRE(t.is_empty());
}

TEST_CASE("Tensor shape+dtype constructor", "[core]") {
    Tensor t({1, 3, 224, 224}, DataType::FP32);
    REQUIRE(t.shape() == std::vector<int64_t>({1, 3, 224, 224}));
    REQUIRE(t.size() == 1 * 3 * 224 * 224);
    REQUIRE(t.byte_size() == 1 * 3 * 224 * 224 * 4);
    REQUIRE(t.dtype() == DataType::FP32);
    REQUIRE(t.device() == Device::CPU);
    REQUIRE(t.data() != nullptr);
    REQUIRE_FALSE(t.is_empty());
}

TEST_CASE("Tensor named constructor", "[core]") {
    Tensor t({10}, DataType::INT32, Device::CPU, "input");
    REQUIRE(t.get_name() == "input");
    t.set_name("output");
    REQUIRE(t.get_name() == "output");
}

// ============ Tensor dtype variants ============

TEST_CASE("Tensor all dtype allocations", "[core]") {
    auto test_dtype = [](DataType dt, size_t elem_size) {
        Tensor t({4}, dt);
        REQUIRE(t.dtype() == dt);
        REQUIRE(t.byte_size() == 4 * elem_size);
        REQUIRE(t.data() != nullptr);
    };
    test_dtype(DataType::FP32, 4);
    test_dtype(DataType::FP64, 8);
    test_dtype(DataType::INT8, 1);
    test_dtype(DataType::INT32, 4);
    test_dtype(DataType::INT64, 8);
    test_dtype(DataType::UINT8, 1);
}

// ============ Tensor data access ============

TEST_CASE("Tensor data read/write", "[core]") {
    Tensor t({10}, DataType::FP32);
    auto* data = static_cast<float*>(t.data());
    for (int i = 0; i < 10; ++i) data[i] = static_cast<float>(i * 1.5f);

    SECTION("read back via data()") {
        auto* rd = static_cast<const float*>(t.data());
        for (int i = 0; i < 10; ++i)
            REQUIRE(rd[i] == i * 1.5f);
    }

    SECTION("read back via at()") {
        for (int i = 0; i < 10; ++i)
            REQUIRE(t.at({i}) == i * 1.5f);
    }
}

TEST_CASE("Tensor UINT8 access", "[core]") {
    Tensor t({16}, DataType::UINT8);
    auto* d = static_cast<uint8_t*>(t.data());
    for (int i = 0; i < 16; ++i) d[i] = static_cast<uint8_t>(i * 17);
    for (int i = 0; i < 16; ++i)
        REQUIRE(static_cast<int>(d[i]) == (i * 17) % 256);
}

// ============ Tensor copy / move ============

TEST_CASE("Tensor copy constructor", "[core]") {
    Tensor a({2, 3}, DataType::FP32);
    auto* da = static_cast<float*>(a.data());
    for (int i = 0; i < 6; ++i) da[i] = static_cast<float>(i);

    Tensor b(a);
    REQUIRE(b.shape() == a.shape());
    REQUIRE(b.dtype() == a.dtype());
    auto* db = static_cast<float*>(b.data());
    for (int i = 0; i < 6; ++i) REQUIRE(db[i] == static_cast<float>(i));

    // Modify original — copy should be independent
    da[0] = 999.0f;
    REQUIRE(db[0] == 0.0f);
}

TEST_CASE("Tensor move constructor", "[core]") {
    Tensor a({5}, DataType::FP64);
    auto* da = static_cast<double*>(a.data());
    for (int i = 0; i < 5; ++i) da[i] = i * 100.0;

    Tensor b(std::move(a));
    REQUIRE(b.shape() == std::vector<int64_t>({5}));
    REQUIRE(b.dtype() == DataType::FP64);
    auto* db = static_cast<double*>(b.data());
    REQUIRE(db[3] == 300.0);
}

TEST_CASE("Tensor copy assignment", "[core]") {
    Tensor a({4, 4}, DataType::FP32);
    Tensor b;
    b = a;
    REQUIRE(b.shape() == a.shape());
    REQUIRE(b.dtype() == a.dtype());
    REQUIRE(b.data() != nullptr);
    REQUIRE(b.data() != a.data());
}

TEST_CASE("Tensor move assignment", "[core]") {
    Tensor a({3, 3}, DataType::FP32);
    void* old_ptr = a.data();
    Tensor b;
    b = std::move(a);
    REQUIRE(b.data() == old_ptr);
}

// ============ Tensor reshape / view ============

TEST_CASE("Tensor reshape view", "[core]") {
    Tensor t({2, 3, 4}, DataType::FP32);
    auto v = t.reshape({6, 4});
    REQUIRE(v.shape() == std::vector<int64_t>({6, 4}));
    REQUIRE(v.data() == t.data());
    REQUIRE(v.size() == 24);
    REQUIRE(v.byte_size() == 24 * 4);
}

TEST_CASE("Tensor transpose view", "[core]") {
    Tensor t({2, 3}, DataType::FP32);
    auto* d = static_cast<float*>(t.data());
    for (int i = 0; i < 6; ++i) d[i] = static_cast<float>(i);

    auto v = t.transpose({1, 0});
    REQUIRE(v.shape() == std::vector<int64_t>({3, 2}));
    REQUIRE_FALSE(v.is_contiguous());
}

TEST_CASE("Tensor slice view", "[core]") {
    Tensor t({4, 4}, DataType::FP32);
    auto* d = static_cast<float*>(t.data());
    for (int i = 0; i < 16; ++i) d[i] = static_cast<float>(i);

    auto v = t.slice({1, 1}, {3, 3});
    REQUIRE(v.shape() == std::vector<int64_t>({2, 2}));
}

TEST_CASE("Tensor view to_tensor", "[core]") {
    Tensor t({3, 4}, DataType::FP32);
    auto* d = static_cast<float*>(t.data());
    for (int i = 0; i < 12; ++i) d[i] = static_cast<float>(i);

    auto v = t.transpose({1, 0});  // now 4x3
    auto m = v.to_tensor();
    REQUIRE(m.shape() == std::vector<int64_t>({4, 3}));
    REQUIRE(m.at({1, 2}) == 9.0f); // t[2][1] = 2*4+1 = 9
}

// ============ Tensor utility methods ============

TEST_CASE("Tensor clone", "[core]") {
    Tensor t({2, 2}, DataType::FP32);
    auto* d = static_cast<float*>(t.data());
    d[0] = 1; d[1] = 2; d[2] = 3; d[3] = 4;

    auto c = t.clone();
    REQUIRE(c.shape() == t.shape());
    REQUIRE(c.data() != t.data());
    auto* cd = static_cast<float*>(c.data());
    REQUIRE(cd[0] == 1.0f);
    REQUIRE(cd[3] == 4.0f);
}

TEST_CASE("Tensor resize", "[core]") {
    Tensor t({1, 1}, DataType::FP32);
    t.resize({2, 3, 4}, DataType::INT32);
    REQUIRE(t.shape() == std::vector<int64_t>({2, 3, 4}));
    REQUIRE(t.dtype() == DataType::INT32);
    REQUIRE(t.byte_size() == 2 * 3 * 4 * 4);
}

TEST_CASE("Tensor allocate after default", "[core]") {
    Tensor t;
    t.allocate({100}, DataType::FP64);
    REQUIRE(t.size() == 100);
    REQUIRE(t.byte_size() == 100 * 8);
    REQUIRE(t.data() != nullptr);
}

TEST_CASE("Tensor external memory", "[core]") {
    std::vector<float> buf(16, 3.14f);
    Tensor t(buf.data(), {4, 4}, DataType::FP32, Device::CPU, nullptr, "ext");
    REQUIRE(t.shape() == std::vector<int64_t>({4, 4}));
    REQUIRE(t.get_name() == "ext");
    // data is accessible and contains expected values
    REQUIRE(static_cast<float*>(t.data())[0] == 3.14f);
    REQUIRE(static_cast<float*>(t.data())[15] == 3.14f);
}

TEST_CASE("Tensor element_size utility", "[core]") {
    REQUIRE(Tensor::get_element_size(DataType::FP32) == 4);
    REQUIRE(Tensor::get_element_size(DataType::FP64) == 8);
    REQUIRE(Tensor::get_element_size(DataType::INT8) == 1);
    REQUIRE(Tensor::get_element_size(DataType::INT32) == 4);
    REQUIRE(Tensor::get_element_size(DataType::INT64) == 8);
    REQUIRE(Tensor::get_element_size(DataType::UINT8) == 1);
}

TEST_CASE("Tensor rank and dim helpers", "[core]") {
    Tensor t({1, 3, 64, 64}, DataType::FP32);
    REQUIRE(t.get_rank() == 4);
    REQUIRE(t.get_dim_size(0) == 1);
    REQUIRE(t.get_dim_size(1) == 3);
    REQUIRE(t.get_dim_size(2) == 64);
    REQUIRE(t.get_dim_size(3) == 64);
    REQUIRE(t.outer_dim(0) == 1);
    REQUIRE(t.outer_dim(1) == 1);
    REQUIRE(t.outer_dim(2) == 3);
    REQUIRE(t.outer_dim(3) == 192);
}

// ============ Tensor edge cases ============

TEST_CASE("Tensor concat", "[core]") {
    Tensor a({2, 3}, DataType::FP32);
    Tensor b({2, 3}, DataType::FP32);
    auto* da = static_cast<float*>(a.data());
    auto* db = static_cast<float*>(b.data());
    for (int i = 0; i < 6; ++i) { da[i] = static_cast<float>(i); db[i] = static_cast<float>(i + 100); }

    auto c = Tensor::concat({a, b}, 0);
    REQUIRE(c.shape() == std::vector<int64_t>({4, 3}));
    REQUIRE(c.at({0, 0}) == 0.0f);
    REQUIRE(c.at({3, 2}) == 105.0f);
}

// ============ TensorView tests ============

TEST_CASE("TensorView from tensor", "[core]") {
    Tensor t({2, 3, 4}, DataType::FP32);
    auto v = t.view();
    REQUIRE(v.shape() == t.shape());
    REQUIRE(v.data() == t.data());
    REQUIRE(v.size() == 24);
}

TEST_CASE("TensorView is_contiguous", "[core]") {
    Tensor t({3, 4}, DataType::FP32);
    REQUIRE(t.view().is_contiguous());
    auto v = t.transpose({1, 0});
    REQUIRE_FALSE(v.is_contiguous());
}

TEST_CASE("TensorView to_tensor", "[core]") {
    Tensor t({2, 2}, DataType::FP32);
    static_cast<float*>(t.data())[0] = 42.0f;
    auto v = t.view();
    auto c = v.to_tensor();
    REQUIRE(c.at({0, 0}) == 42.0f);
    REQUIRE(c.data() != t.data());  // to_tensor should materialize
}

// ============ to_string / printing ============

TEST_CASE("Tensor to_string", "[core]") {
    Tensor t({2, 2}, DataType::FP32);
    auto* d = static_cast<float*>(t.data());
    d[0] = 1; d[1] = 2; d[2] = 3; d[3] = 4;

    auto s = t.to_string();
    REQUIRE_FALSE(s.empty());
}

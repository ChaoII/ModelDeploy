#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include "core/tensor.h"
#include "runtime/runtime_option.h"
#include "runtime/runtime.h"

using namespace modeldeploy;

namespace {
std::filesystem::path write_min_ncnn() {
    auto dir = std::filesystem::temp_directory_path() / "md_ncnn_test";
    std::filesystem::create_directories(dir);
    auto param = dir / "min.param";
    {
        std::ofstream ofs(param);
        ofs << "7767517\n";
        ofs << "2 2\n";
        ofs << "Input  input  0 1 data  0=1 1=3 2=64 3=64\n";
        ofs << "ReLU   relu   1 1 data relu\n";
    }
    {
        std::ofstream ofs(dir / "min.bin", std::ios::binary);
    }
    return param;
}
}  // namespace

TEST_CASE("ncnn backend CPU infer", "[ncnn][cpu]") {
#ifdef ENABLE_NCNN
    auto param = write_min_ncnn();
    RuntimeOption opt;
    opt.use_ncnn_backend();
    opt.set_device(Device::CPU, 0);
    opt.set_model_path(param.string());
    Runtime rt;
    REQUIRE(rt.init(opt));
    REQUIRE(rt.num_inputs() == 1);
    REQUIRE(rt.num_outputs() == 1);

    Tensor input({1, 3, 64, 64}, DataType::FP32, Device::CPU, "data");
    for (int i = 0; i < static_cast<int>(input.byte_size() / sizeof(float)); ++i) {
        reinterpret_cast<float*>(input.data())[i] = 1.0f;
    }
    std::vector<Tensor> ins{input};
    std::vector<Tensor> outs;
    REQUIRE(rt.infer(ins, &outs));
    REQUIRE(outs.size() == 1);
    auto shp = outs[0].shape();
    REQUIRE(shp.size() == 4);
    REQUIRE((shp[0] == 1 && shp[1] == 3 && shp[2] == 64 && shp[3] == 64));
    // ReLU 恒等：输入全 1 → 输出全 1
    auto* od = reinterpret_cast<float*>(outs[0].data());
    REQUIRE(od[0] == Catch::Approx(1.0f).margin(1e-3f));
#else
    WARN("ENABLE_NCNN off, skipping");
#endif
}

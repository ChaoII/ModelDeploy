#include <catch2/catch_test_macros.hpp>
#include "runtime/runtime_option.h"

using namespace modeldeploy;

TEST_CASE("RuntimeOption MNN maps OPENCL/VULKAN to forward_type", "[core]") {
    RuntimeOption opencl_opt;
    opencl_opt.use_mnn_backend();
    opencl_opt.set_device(Device::OPENCL, 0);
    opencl_opt.validate();
    REQUIRE(opencl_opt.mnn_option.forward_type ==
            modeldeploy::mnn::MNN_FORWARD_OPENCL);

    RuntimeOption vulkan_opt;
    vulkan_opt.use_mnn_backend();
    vulkan_opt.set_device(Device::VULKAN, 0);
    vulkan_opt.validate();
    REQUIRE(vulkan_opt.mnn_option.forward_type ==
            modeldeploy::mnn::MNN_FORWARD_VULKAN);
}

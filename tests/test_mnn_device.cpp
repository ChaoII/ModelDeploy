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

// ==================== 设备集成冒烟（标签隔离） ====================
// 需要真实 test_data（含 mnn/yolo26n/yolo26n-cls.mnn + test_images）+ OpenCL/Vulkan 运行时。
// 无设备/无数据的机器上优雅跳过（沿用 baseline_compare 的 exists 守卫），CPU-only CI 用
// [opencl]/[vulkan] 标签排除，不会默认运行。
#ifdef ENABLE_MNN
#include <filesystem>
#include "vision/classification/classification.h"

using namespace modeldeploy::vision;

namespace {

std::filesystem::path smoke_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return std::filesystem::path(env) / "test_data";
    return std::filesystem::current_path() / "test_data";
}

std::filesystem::path smoke_model(const std::string& rel, const std::string& backend) {
    return smoke_test_data() / "test_models" / backend / rel;
}

std::filesystem::path smoke_image(const std::string& name) {
    return smoke_test_data() / "test_images" / name;
}

}  // namespace

TEST_CASE("MNN OpenCL smoke", "[opencl]") {
    auto model = smoke_model("yolo26n/yolo26n-cls.mnn", "mnn");
    if (!std::filesystem::exists(model)) return;
    auto imgf = smoke_image("test_person.jpg");
    if (!std::filesystem::exists(imgf)) return;

    RuntimeOption opt;
    opt.use_mnn_backend();
    opt.set_device(Device::OPENCL, 0);
    modeldeploy::vision::classification::Classification cls(model.string(), opt);
    REQUIRE(cls.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult res;
    REQUIRE(cls.predict(img, &res));
    REQUIRE_FALSE(res.label_ids.empty());
}

TEST_CASE("MNN Vulkan smoke", "[vulkan]") {
    auto model = smoke_model("yolo26n/yolo26n-cls.mnn", "mnn");
    if (!std::filesystem::exists(model)) return;
    auto imgf = smoke_image("test_person.jpg");
    if (!std::filesystem::exists(imgf)) return;

    RuntimeOption opt;
    opt.use_mnn_backend();
    opt.set_device(Device::VULKAN, 0);
    modeldeploy::vision::classification::Classification cls(model.string(), opt);
    REQUIRE(cls.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult res;
    REQUIRE(cls.predict(img, &res));
    REQUIRE_FALSE(res.label_ids.empty());
}
#endif  // ENABLE_MNN


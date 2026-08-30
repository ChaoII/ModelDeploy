#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>
#include "encryption/encryption.h"
#include "runtime/runtime_option.h"
#include "vision/sam/fastsam.h"

namespace fs = std::filesystem;

namespace {
    fs::path e2e_test_data_path() {
        const char* env = std::getenv("TEST_DATA_DIR");
        if (env && *env) return fs::path(env) / "test_data";
        return fs::current_path() / "test_data";
    }

    // 栈上 RAII 清理守卫：用例结束/失败/异常时都会删除临时文件，避免系统临时目录残留。
    struct E2eFileGuard {
        fs::path onnx_path;
        fs::path mdenc_path;
        ~E2eFileGuard() {
            std::error_code ec;
            fs::remove(onnx_path, ec);
            fs::remove(mdenc_path, ec);
        }
    };
} // namespace

// 真实 onnx 端到端：运行时拷贝 -> 加密 -> RuntimeOption 自动解密加载 -> predict -> 错误密码拒绝。
// 仅在 ENABLE_ENCRYPTION 下运行；未启用时 no-op（SKIP）。
#ifdef ENABLE_ENCRYPTION
TEST_CASE("Encryption e2e load encrypted onnx", "[encryption]") {
    const fs::path src_model = e2e_test_data_path() / "test_models" / "onnx" / "FastSAM-s.onnx";
    const fs::path img_path = e2e_test_data_path() / "test_images" / "test_detection0.jpg";
    if (!fs::exists(src_model)) {
        SKIP("FastSAM-s.onnx 测试模型缺失，跳过加密 e2e 验证");
    }
    if (!fs::exists(img_path)) {
        SKIP("测试图片缺失，跳过加密 e2e 验证");
    }

    const fs::path tmp_dir = fs::temp_directory_path();
    const fs::path e2e_onnx = tmp_dir / "e2e.onnx";
    const fs::path e2e_mdenc = tmp_dir / "e2e.mdenc";
    const std::string password = "e2e_pass";

    // 预删除上次运行的残留，再挂载 RAII 清理守卫（任何路径/异常都会释放临时文件）
    fs::remove(e2e_onnx);
    fs::remove(e2e_mdenc);
    E2eFileGuard guard{e2e_onnx, e2e_mdenc};

    // 1. 拷贝真实 onnx 到临时文件并加密
    REQUIRE(fs::copy_file(src_model, e2e_onnx, fs::copy_options::overwrite_existing));
    REQUIRE(fs::file_size(e2e_onnx) > 0);
    REQUIRE(modeldeploy::encrypt_model_file(e2e_onnx.string(), e2e_mdenc.string(), password, "onnx"));
    REQUIRE(modeldeploy::is_encrypted_model_file(e2e_mdenc.string()));
    REQUIRE(fs::file_size(e2e_mdenc) > 0);

    // 2. 正例：set_model_path(mdenc, password) 自动解密，构造成功。
    //    FastSam 构造内部会对同一加密文件以空密码再次 set_model_path，若 RuntimeOption::password
    //    为空则 fallback 解密（runtime_option.cpp:41-42）拿到空密码而抛异常；故密码必须同时
    //    通过 set_password 写入 option，随拷贝保留供二次加载回退。
    modeldeploy::RuntimeOption opt;
    opt.set_password(password);
    opt.set_model_path(e2e_mdenc.string(), password);
    modeldeploy::vision::seg::FastSam sam(e2e_mdenc.string(), opt);
    REQUIRE(sam.is_initialized());

    // 3. 用测试图 predict 一次
    auto img = modeldeploy::vision::ImageData::imread(img_path.string());
    REQUIRE_FALSE(img.empty());
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    REQUIRE(sam.predict(img, &res));
    REQUIRE_FALSE(res.empty());

    // 4. 负例：错误密码应构造失败（set_model_path 解密失败抛异常，或 is_initialized false）
    bool rejected = false;
    try {
        modeldeploy::vision::seg::FastSam bad(e2e_mdenc.string(), modeldeploy::RuntimeOption{});
        rejected = !bad.is_initialized();
    } catch (const std::exception&) {
        rejected = true;
    }
    REQUIRE(rejected);
}
#else
TEST_CASE("Encryption e2e load encrypted onnx", "[encryption]") {
    SKIP("ENABLE_ENCRYPTION 未启用，跳过加密 e2e 验证");
}
#endif

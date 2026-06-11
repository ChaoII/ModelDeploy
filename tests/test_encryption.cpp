#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <cstring>
#include "encryption/encryption.h"

namespace fs = std::filesystem;

TEST_CASE("Encryption roundtrip file", "[encryption]") {
    // 创建一个测试 ONNX 文件（fake 内容）
    auto tmp_dir = fs::temp_directory_path();
    auto input = tmp_dir / "test_encrypt_model.onnx";
    auto encrypted = tmp_dir / "test_encrypt_model.mdenc";
    auto decrypted = tmp_dir / "test_encrypt_decoded.onnx";

    // 写入测试数据
    {
        std::ofstream ofs(input, std::ios::binary);
        ofs << "THIS IS A FAKE ONNX MODEL DATA\0\xff\xfe\x01";
    }

    // 加密
    REQUIRE(modeldeploy::encrypt_model_file(input.string(), encrypted.string(), "test123", "onnx"));

    // 加密文件应能识别为加密模型
    REQUIRE(modeldeploy::is_encrypted_model_file(encrypted.string()));
    // 原始 ONNX 不应被识别为加密
    REQUIRE_FALSE(modeldeploy::is_encrypted_model_file(input.string()));

    // 格式检测
    auto format = modeldeploy::get_model_format_from_encrypted_file(encrypted.string());
    REQUIRE(format == "onnx");

    // 解密
    REQUIRE(modeldeploy::decrypt_model_file(encrypted.string(), decrypted.string(), "test123"));

    // 解密后内容应与原始一致
    {
        std::ifstream ifs(decrypted, std::ios::binary);
        std::string content((std::istreambuf_iterator(ifs)), std::istreambuf_iterator<char>());
        REQUIRE(content.find("THIS IS A FAKE ONNX MODEL DATA") != std::string::npos);
    }

    // 错误密码解密应失败
    REQUIRE_FALSE(modeldeploy::decrypt_model_file(encrypted.string(), decrypted.string(), "wrong_password"));

    // 清理
    fs::remove(input);
    fs::remove(encrypted);
    fs::remove(decrypted);
}

TEST_CASE("Encryption roundtrip buffer", "[encryption]") {
    auto tmp_dir = fs::temp_directory_path();
    auto input = tmp_dir / "test_encrypt_buf.onnx";
    auto encrypted = tmp_dir / "test_encrypt_buf.mdenc";

    // 写入测试数据
    {
        std::ofstream ofs(input, std::ios::binary);
        ofs << "MODEL_BUFFER_DATA_12345\x00\x01\x02";
    }

    // 加密
    REQUIRE(modeldeploy::encrypt_model_file(input.string(), encrypted.string(), "p@ssw0rd", "onnx"));

    // 读取加密模型到内存 buffer
    std::string buffer;
    std::string format;
    REQUIRE(modeldeploy::read_encrypted_model_to_buffer(encrypted.string(), "p@ssw0rd", &buffer, &format));
    REQUIRE(format == "onnx");
    REQUIRE(buffer.find("MODEL_BUFFER_DATA_12345") != std::string::npos);

    // 错误密码应失败
    std::string bad_buf, bad_fmt;
    REQUIRE_FALSE(modeldeploy::read_encrypted_model_to_buffer(encrypted.string(), "wrong", &bad_buf, &bad_fmt));

    fs::remove(input);
    fs::remove(encrypted);
}

TEST_CASE("Encryption empty password rejected", "[encryption]") {
    auto tmp_dir = fs::temp_directory_path();
    auto input = tmp_dir / "test_empty_pwd.onnx";
    auto encrypted = tmp_dir / "test_empty_pwd.mdenc";

    {
        std::ofstream ofs(input, std::ios::binary);
        ofs << "DATA";
    }

    // 空密码应被拒绝（AES-256 不支持空密码）
    REQUIRE_FALSE(modeldeploy::encrypt_model_file(input.string(), encrypted.string(), "", "onnx"));

    fs::remove(input);
    fs::remove(encrypted);
}

TEST_CASE("Encryption long password", "[encryption]") {
    auto tmp_dir = fs::temp_directory_path();
    auto input = tmp_dir / "test_long_pwd.onnx";
    auto encrypted = tmp_dir / "test_long_pwd.mdenc";

    {
        std::ofstream ofs(input, std::ios::binary);
        ofs << "LONG_PASSWORD_TEST_DATA";
    }

    std::string long_pwd(128, 'A');
    REQUIRE(modeldeploy::encrypt_model_file(input.string(), encrypted.string(), long_pwd, "onnx"));
    REQUIRE(modeldeploy::is_encrypted_model_file(encrypted.string()));

    auto decrypted = tmp_dir / "test_long_pwd_decoded.onnx";
    REQUIRE(modeldeploy::decrypt_model_file(encrypted.string(), decrypted.string(), long_pwd));

    {
        std::ifstream ifs(decrypted);
        std::string content((std::istreambuf_iterator(ifs)), std::istreambuf_iterator<char>());
        REQUIRE(content == "LONG_PASSWORD_TEST_DATA");
    }

    fs::remove(input);
    fs::remove(encrypted);
    fs::remove(decrypted);
}

TEST_CASE("Encryption CRC mismatch detection", "[encryption]") {
    auto tmp_dir = fs::temp_directory_path();
    auto input = tmp_dir / "test_crc.onnx";
    auto encrypted = tmp_dir / "test_crc.mdenc";

    {
        std::ofstream ofs(input, std::ios::binary);
        ofs << "CRC_CHECK_DATA";
    }

    REQUIRE(modeldeploy::encrypt_model_file(input.string(), encrypted.string(), "crc_test", "onnx"));

    // 篡改加密文件内容，CRC 应不匹配导致解密失败
    auto decrypted = tmp_dir / "test_crc_decoded.onnx";
    {
        std::fstream f(encrypted, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0, std::ios::end);
        auto pos = f.tellp();
        if (pos > 10) {
            f.seekp(-5, std::ios::end);
            f.put('X');
        }
    }
    // 应该解密失败（CRC 不匹配）
    REQUIRE_FALSE(modeldeploy::decrypt_model_file(encrypted.string(), decrypted.string(), "crc_test"));

    fs::remove(input);
    fs::remove(encrypted);
}

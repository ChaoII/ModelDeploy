#pragma once
#include <string>
#include <cstdint>
#include "core/md_decl.h"

namespace modeldeploy {
#ifdef ENABLE_ENCRYPTION
    uint32_t calculate_crc32(const std::string& data);
    MODELDEPLOY_CXX_EXPORT bool encrypt_model_file(const std::string& input_path, const std::string& output_path,
                                                   const std::string& password, const std::string& model_format);
    MODELDEPLOY_CXX_EXPORT bool decrypt_model_file(const std::string& input_path, const std::string& output_path,
                                                   const std::string& password);
    MODELDEPLOY_CXX_EXPORT bool is_encrypted_model_file(const std::string& file_path);
    MODELDEPLOY_CXX_EXPORT std::string get_model_format_from_encrypted_file(const std::string& file_path);
    MODELDEPLOY_CXX_EXPORT bool read_encrypted_model_to_buffer(const std::string& file_path,
                                                               const std::string& password,
                                                               std::string* model_buffer, std::string* model_format);
#else
    // 加密未启用时，所有函数返回默认值
    inline uint32_t calculate_crc32(const std::string&) { return 0; }
    inline bool encrypt_model_file(const std::string&, const std::string&, const std::string&, const std::string&) { return false; }
    inline bool decrypt_model_file(const std::string&, const std::string&, const std::string&) { return false; }
    inline bool is_encrypted_model_file(const std::string&) { return false; }
    inline std::string get_model_format_from_encrypted_file(const std::string&) { return ""; }
    inline bool read_encrypted_model_to_buffer(const std::string&, const std::string&, std::string*, std::string*) { return false; }
#endif
}

//
// Created by aichao on 2025/7/17.
//

#pragma once
#include <string>
#include "core/md_decl.h"

namespace modeldeploy {
    // 模型加密相关函数

    // 计算原始数据的CRC32校验和
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
}

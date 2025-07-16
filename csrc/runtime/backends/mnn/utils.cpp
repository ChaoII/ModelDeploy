//
// Created by aichao on 2025/2/19.
//

#include "core/md_log.h"
#include "runtime/backends/mnn/utils.h"

#include <unordered_map>


namespace modeldeploy {
    halide_type_t md_dtype_to_mnn_dtype(const DataType& dtype) {
        if (dtype == DataType::FP32) {
            return halide_type_of<float>();
        }
        if (dtype == DataType::FP64) {
            return halide_type_of<double>();
        }
        if (dtype == DataType::INT8) {
            return halide_type_of<int8_t>();
        }
        if (dtype == DataType::INT32) {
            return halide_type_of<int32_t>();
        }
        if (dtype == DataType::INT64) {
            return halide_type_of<int64_t>();
        }
        if (dtype == DataType::UINT8) {
            return halide_type_of<uint8_t>();
        }
        MD_LOG_ERROR << "MNN tensor type don't support this type" << std::endl;
        return halide_type_of<float>();
    }

    DataType mnn_dtype_to_md_dtype(const halide_type_t& mnn_dtype) {
        if (mnn_dtype == halide_type_of<float>()) {
            return DataType::FP32;
        }
        if (mnn_dtype == halide_type_of<double>()) {
            return DataType::FP64;
        }
        if (mnn_dtype == halide_type_of<int8_t>()) {
            return DataType::INT8;
        }
        if (mnn_dtype == halide_type_of<int32_t>()) {
            return DataType::INT32;
        }
        if (mnn_dtype == halide_type_of<int64_t>()) {
            return DataType::INT64;
        }
        if (mnn_dtype == halide_type_of<uint8_t>()) {
            return DataType::UINT8;
        }
        MD_LOG_ERROR << "ModelDeploy DataType don't support this type" << std::endl;
        return DataType::UNKNOWN;
    }


    std::string mnn_type_to_string(const halide_type_t& type) {
        // const static std::unordered_map<halide_type_t, std::string> type_map = {
        //     {halide_type_of<float>(), "FLOAT"},
        //     {halide_type_of<int32_t>(), "INT32"},
        //     {halide_type_of<double>(), "DOUBLE"},
        //     {halide_type_of<uint8_t>(), "UINT8"},
        //     {halide_type_of<int8_t>(), "INT8"},
        //     {halide_type_of<int64_t>(), "INT64"}
        //     // 添加更多类型映射...
        // };
        // return type_map.contains(type) ? type_map.at(type) : "UNKNOWN";
        return "";
    }
}

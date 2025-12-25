#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "core/tensor.h"
#include "utils/utils.h"


namespace modeldeploy {
    struct InferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            delete obj;
        }
    };

    // TensorRT中计算张量元素总量的方法
    int64_t volume(const nvinfer1::Dims& d);

    nvinfer1::Dims vec_to_dims(const std::vector<int>& vec);

    nvinfer1::Dims vec_to_dims(const std::vector<int64_t>& vec);

    size_t trt_data_type_size(const nvinfer1::DataType& dtype);

    DataType trt_dtype_to_md_dtype(const nvinfer1::DataType& dtype);

    std::vector<int> dims_to_vec(const nvinfer1::Dims& dim);


    class MDTrtLogger : public nvinfer1::ILogger {
    public:
        static MDTrtLogger* logger;

        static MDTrtLogger* get() {
            if (logger != nullptr) {
                return logger;
            }
            logger = new MDTrtLogger();
            return logger;
        }

        void set_log(const bool enable_info = false, const bool enable_warning = false) {
            enable_info_ = enable_info;
            enable_warning_ = enable_warning;
        }

        void log(nvinfer1::ILogger::Severity severity,
                 const char* msg) noexcept override {
            if (severity == nvinfer1::ILogger::Severity::kINFO) {
                if (enable_info_) {
                    MD_LOG_INFO << msg << std::endl;
                }
            }
            else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
                if (enable_warning_) {
                    MD_LOG_INFO << msg << std::endl;
                }
            }
            else if (severity == nvinfer1::ILogger::Severity::kERROR) {
                MD_LOG_ERROR << msg << std::endl;
            }
            else if (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR) {
                MD_LOG_FATAL << msg << std::endl;
            }
        }

    private:
        bool enable_info_ = false;
        bool enable_warning_ = false;
    };
} // namespace modeldeploy

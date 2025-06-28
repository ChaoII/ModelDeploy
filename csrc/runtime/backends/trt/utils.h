#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "csrc/core/tensor.h"
#include "csrc/utils/utils.h"


namespace modeldeploy {
    struct FDInferDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) {
                delete obj;
                //      obj->destroy();
            }
        }
    };

    template <typename T>
    using FDUniquePtr = std::unique_ptr<T, FDInferDeleter>;

    int64_t Volume(const nvinfer1::Dims& d);

    nvinfer1::Dims ToDims(const std::vector<int>& vec);
    nvinfer1::Dims ToDims(const std::vector<int64_t>& vec);

    size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);

    DataType GetFDDataType(const nvinfer1::DataType& dtype);

    nvinfer1::DataType ReaderDtypeToTrtDtype(int reader_dtype);

    DataType ReaderDtypeToFDDtype(int reader_dtype);

    std::vector<int> ToVec(const nvinfer1::Dims& dim);

    template <typename T>
    std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
        out << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != vec.size() - 1) {
                out << vec[i] << ", ";
            }
            else {
                out << vec[i] << "]";
            }
        }
        return out;
    }

    class FDTrtLogger : public nvinfer1::ILogger {
    public:
        static FDTrtLogger* logger;

        static FDTrtLogger* Get() {
            if (logger != nullptr) {
                return logger;
            }
            logger = new FDTrtLogger();
            return logger;
        }

        void SetLog(bool enable_info = false, bool enable_warning = false) {
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

    struct ShapeRangeInfo {
        explicit ShapeRangeInfo(const std::vector<int64_t>& new_shape) {
            shape.assign(new_shape.begin(), new_shape.end());
            min.resize(new_shape.size());
            max.resize(new_shape.size());
            is_static.resize(new_shape.size());
            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (new_shape[i] > 0) {
                    min[i] = new_shape[i];
                    max[i] = new_shape[i];
                    is_static[i] = 1;
                }
                else {
                    min[i] = -1;
                    max[i] = -1;
                    is_static[i] = 0;
                }
            }
        }

        std::string name;
        std::vector<int64_t> shape;
        std::vector<int64_t> min;
        std::vector<int64_t> max;
        std::vector<int64_t> opt;
        std::vector<int8_t> is_static;
        // return
        // -1: new shape is inillegal
        // 0 : new shape is able to inference
        // 1 : new shape is out of range, need to update engine
        int Update(const std::vector<int64_t>& new_shape);

        int Update(const std::vector<int>& new_shape) {
            std::vector<int64_t> new_shape_int64(new_shape.begin(), new_shape.end());
            return Update(new_shape_int64);
        }

        friend std::ostream& operator<<(std::ostream& out,
                                        const ShapeRangeInfo& info) {
            out << "Input name: " << info.name << ", shape=" << info.shape
                << ", min=" << info.min << ", max=" << info.max << std::endl;
            return out;
        }
    };
} // namespace fastdeploy

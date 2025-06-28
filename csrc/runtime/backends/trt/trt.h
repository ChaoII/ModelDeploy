//
// Created by aichao on 2025/6/27.
//

#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "csrc/runtime/backends/backend.h"
#include "csrc/runtime/backends/trt/utils.h"
#include "csrc/runtime/backends/trt/option.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit Int8EntropyCalibrator2(const std::string& calibration_cache)
        : calibration_cache_(calibration_cache) {
    }

    int getBatchSize() const noexcept override { return 0; }

    bool getBatch(void* bindings[], const char* names[],
                  int nbBindings) noexcept override {
        return false;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        length = calibration_cache_.size();
        return length ? calibration_cache_.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache,
                               size_t length) noexcept override {
        MD_LOG_ERROR << "NOT IMPLEMENT." << std::endl;
    }

private:
    const std::string calibration_cache_;
};

namespace modeldeploy {
    struct TrtValueInfo {
        std::string name;
        std::vector<int> shape;
        nvinfer1::DataType dtype; // dtype of TRT model
        DataType original_dtype; // dtype of original ONNX/Paddle model
    };

    std::vector<int> toVec(const nvinfer1::Dims& dim);
    size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);
    DataType GetFDDataType(const nvinfer1::DataType& dtype);

    class TrtBackend : public BaseBackend {
    public:
        TrtBackend() : engine_(nullptr), context_(nullptr) {
        }

        bool init(const RuntimeOption& runtime_option);
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        size_t num_inputs() const override { return inputs_desc_.size(); }
        size_t num_outputs() const override { return outputs_desc_.size(); }
        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        std::unique_ptr<BaseBackend> clone(RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

        ~TrtBackend() {
            if (parser_) {
                parser_.reset();
            }
        }

    private:
        void BuildOption(const TrtBackendOption& option);
        bool init_from_onnx(const std::string& model_buffer,
                            const TrtBackendOption& option = TrtBackendOption());

        TrtBackendOption option_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<nvonnxparser::IParser> parser_;
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::IBuilder> builder_;
        std::unique_ptr<nvinfer1::INetworkDefinition> network_;
        cudaStream_t stream_{};
        std::vector<void*> bindings_;
        std::vector<TrtValueInfo> inputs_desc_;
        std::vector<TrtValueInfo> outputs_desc_;
        std::map<std::string, int> io_name_index_;

        std::string model_buffer_;
        std::string calibration_str_;
        bool save_external_ = false;
        std::string model_file_name_;

        // Sometimes while the number of outputs > 1
        // the output order of tensorrt may not be same
        // with the original onnx model
        // So this parameter will record to origin outputs
        // order, to help recover the rigt order
        std::map<std::string, int> outputs_order_;

        // temporary store onnx model content
        // once it used to build trt egnine done
        // it will be released
        std::string onnx_model_buffer_;
        // Stores shape information of the loaded model
        // For dynmaic shape will record its range information
        // Also will update the range information while inferencing
        std::map<std::string, ShapeRangeInfo> shape_range_info_;

        // If the final output tensor's dtype is different from the
        // model output tensor's dtype, then we need cast the data
        // to the final output's dtype.
        // E.g. When trt model output tensor is int32, but final tensor is int64
        // This map stores the casted tensors.
        std::map<std::string, Tensor> casted_output_tensors_;

        void get_input_output_info();
        bool CreateTrtEngineFromOnnx(const std::string& onnx_model_buffer);
        bool LoadTrtCache(const std::string& trt_engine_file);
        int ShapeRangeInfoUpdated(const std::vector<Tensor>& inputs);
    };
} // namespace fastdeploy

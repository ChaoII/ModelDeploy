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


namespace modeldeploy {
    struct TrtValueInfo {
        std::string name;
        std::vector<int> shape;
        nvinfer1::DataType dtype; // dtype of TRT model
    };


    class TrtBackendImpl {
    public:
        TrtBackendImpl() : engine_(nullptr), context_(nullptr) {
        }

        bool init(const RuntimeOption& runtime_option);
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs);

        [[nodiscard]] size_t num_inputs() const { return inputs_desc_.size(); }
        [[nodiscard]] size_t num_outputs() const { return outputs_desc_.size(); }
        TensorInfo get_input_info(int index);
        TensorInfo get_output_info(int index);
        std::vector<TensorInfo> get_input_infos();
        std::vector<TensorInfo> get_output_infos();
        std::unique_ptr<TrtBackendImpl> clone(RuntimeOption& runtime_option,
                                              void* stream = nullptr,
                                              int device_id = -1);

        ~TrtBackendImpl() {
            if (parser_) {
                parser_.reset();
            }
        }

    private:
        bool init_from_onnx(const std::string& model_buffer);
        bool initialized_ = false;
        TrtBackendOption option_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<nvonnxparser::IParser> parser_;
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::IBuilder> builder_;
        std::unique_ptr<nvinfer1::INetworkDefinition> network_;
        cudaStream_t stream_{};
        std::vector<TrtValueInfo> inputs_desc_;
        std::vector<TrtValueInfo> outputs_desc_;
        std::string model_buffer_;
        std::map<std::string, ShapeRangeInfo> shape_range_info_;

        void get_input_output_info();
        bool create_trt_engine_from_onnx(const std::string& onnx_model_buffer);
        bool load_trt_cache(const std::string& trt_engine_file);
        int shape_range_info_updated(const std::vector<Tensor>& inputs);
    };
} // namespace modeldeploy

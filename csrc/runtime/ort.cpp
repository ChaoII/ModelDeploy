//
// Created by aichao on 2025/2/20.
//

#include "ort.h"
#include "../utils/utils.h"
#include <iostream>
#include <fstream>
namespace modeldeploy {


    void OrtBackend::build_option(const RuntimeOption& option) {
        option_ = option;
        if (option.graph_optimization_level >= 0) {
            session_options_.SetGraphOptimizationLevel(
                static_cast<GraphOptimizationLevel>(option.graph_optimization_level));
        }
        if (option.intra_op_num_threads > 0) {
            session_options_.SetIntraOpNumThreads(option.intra_op_num_threads);
        }
        if (option.inter_op_num_threads > 0) {
            session_options_.SetInterOpNumThreads(option.inter_op_num_threads);
        }
        if (option.execution_mode >= 0) {
            session_options_.SetExecutionMode(static_cast<ExecutionMode>(option.execution_mode));
        }
        if (!option.optimized_model_filepath.empty()) {
#ifdef _WIN32
            session_options_.SetOptimizedModelFilePath(to_wstring(option.optimized_model_filepath).c_str());
#else
            session_options_.SetOptimizedModelFilePath(option.optimized_model_filepath.c_str());
#endif
        }
        // CUDA
        if (option.device == Device::GPU) {
            auto all_providers = Ort::GetAvailableProviders();
            bool support_cuda = false;
            std::string providers_msg;
            for (const auto& all_provider : all_providers) {
                providers_msg += (all_provider + ", ");
                if (all_provider == "CUDAExecutionProvider") {
                    support_cuda = true;
                }
            }
            if (!support_cuda) {
                std::cerr << "Compiled fastdeploy with onnxruntime doesn't "
                    "support GPU, the available providers are "
                    << providers_msg << "will fallback to CPUExecutionProvider."
                    << std::endl;
                option_.device = Device::CPU;
            }
            else {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = option.device_id;
                if (option.external_stream_) {
                    cuda_options.has_user_compute_stream = 1;
                    cuda_options.user_compute_stream = option.external_stream_;
                }
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
            }
        }
    }

    bool OrtBackend::init(const RuntimeOption& option) {
        if (option.device != Device::CPU && option.device != Device::GPU) {
            std::cerr
                << "Backend::ORT only supports Device::CPU/Device::GPU, but now its "
                << option.device << "." << std::endl;
            return false;
        }
        if (option.model_from_memory) {
            return init_from_onnx(option.model_buffer, option);
        }
        std::string model_buffer;
        read_binary_from_file(option.model_filepath, &model_buffer);
        return init_from_onnx(model_buffer, option);
    }


    bool OrtBackend::init_from_onnx(const std::string& model_file, const RuntimeOption& option) {
        if (initialized_) {
            std::cerr << "OrtBackend is already initlized, cannot initialize again." << std::endl;
            return false;
        }
        build_option(option);
        session_ = {env_, model_file.data(), model_file.size(), session_options_};
        binding_ = std::make_unique<Ort::IoBinding>(session_);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Allocator allocator(session_, memory_info);
        std::cout << "======================Input Info===========================" << std::endl;
        size_t n_inputs = session_.GetInputCount();
        for (size_t i = 0; i < n_inputs; ++i) {
            auto input_name_ptr = session_.GetInputNameAllocated(i, allocator);
            auto type_info = session_.GetInputTypeInfo(i);
            std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            ONNXTensorElementDataType data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();

            std::cout << "Input" << i << ":" << " name=" << input_name_ptr.get() << std::endl;
            std::cout << "Input" << i << ":" << " data type=" << data_type << std::endl;
            std::cout << "Input" << i << " : num_dims = " << shape.size() << '\n';
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << "Input" << i << " : dim[" << j << "] =" << shape[j] << '\n';
            }
            inputs_desc_.emplace_back(TensorInfo{input_name_ptr.get(), shape, data_type});
        }
        std::cout << "======================Output Info===========================" << std::endl;
        size_t n_outputs = session_.GetOutputCount();
        for (size_t i = 0; i < n_outputs; ++i) {
            auto output_name_ptr = session_.GetOutputNameAllocated(i, allocator);
            auto type_info = session_.GetOutputTypeInfo(i);
            std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            ONNXTensorElementDataType data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
            outputs_desc_.emplace_back(TensorInfo{output_name_ptr.get(), shape, data_type});
            std::cout << "Output" << i << ":" << " name:" << output_name_ptr.get() << std::endl;
            std::cout << "Output" << i << ":" << " data type: " << data_type << std::endl;
            std::cout << "Output" << i << ": num_dims = " << shape.size() << '\n';
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << "Output" << i << " : " << " dim[" << j << "] =" << shape[j] << '\n';
            }
            Ort::MemoryInfo out_memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
            binding_->BindOutput(output_name_ptr.get(), out_memory_info);
        }
        initialized_ = true;
        return true;
    }

    bool OrtBackend::infer(std::vector<MDTensor>& inputs, std::vector<MDTensor>* outputs, bool copy_to_fd) {
        if (inputs.size() != inputs_desc_.size()) {
            std::cerr << "[OrtBackend] Size of the inputs(" << inputs.size()
                << ") should keep same with the inputs of this model("
                << inputs_desc_.size() << ")." << std::endl;
            return false;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto ort_value = create_ort_value(inputs[i], false);
            binding_->BindInput(inputs_desc_[i].name.c_str(), ort_value);
        }

        for (auto& output_info : outputs_desc_) {
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            binding_->BindOutput(output_info.name.c_str(), memory_info);
        }
        try {
            session_.Run({}, *binding_);
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to Infer: " << e.what() << std::endl;
            return false;
        }
        std::vector<Ort::Value> ort_outputs = binding_->GetOutputValues();
        outputs->resize(ort_outputs.size());
        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            ort_value_to_md_tensor(ort_outputs[i], &((*outputs)[i]), outputs_desc_[i].name,
                               copy_to_fd);
        }
        return true;
    }

    TensorInfo OrtBackend::get_input_info(int index) {
        return inputs_desc_[index];
    }

    std::vector<TensorInfo> OrtBackend::get_input_infos() {
        return inputs_desc_;
    }

    TensorInfo OrtBackend::get_output_info(int index) {
        return outputs_desc_[index];
    }

    std::vector<TensorInfo> OrtBackend::get_output_infos() {
        return outputs_desc_;
    }
}
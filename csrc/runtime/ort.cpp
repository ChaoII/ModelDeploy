//
// Created by aichao on 2025/2/20.
//
#include <iostream>
#include <fstream>
#include <filesystem>
#include <tabulate/tabulate.hpp>
#include "csrc/runtime/ort.h"
#include "csrc/utils/utils.h"
#include "csrc/core/md_log.h"


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
            const auto all_providers = Ort::GetAvailableProviders();
            bool support_cuda = false;
            std::string providers_msg;
            for (const auto& all_provider : all_providers) {
                providers_msg += all_provider + ", ";
                if (all_provider == "CUDAExecutionProvider") {
                    support_cuda = true;
                }
            }
            if (!support_cuda) {
                MD_LOG_ERROR << "Compiled fastdeploy with onnxruntime doesn't support GPU, "
                    "the available providers are " << providers_msg
                    << " will fallback to CPUExecutionProvider." << std::endl;
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
            MD_LOG_ERROR << "Backend::ORT only supports Device::CPU/Device::GPU, but now its "
                << (option.device == 0 ? "cpu" : "gpu") << "." << std::endl;
            return false;
        }
        if (option.model_from_memory) {
            return init_from_onnx(option.model_buffer, option);
        }
        if (!std::filesystem::exists(option.model_filepath)) {
            MD_LOG_ERROR << "Model file does not exist: " << option.model_filepath << std::endl;
            return false;
        }
        std::string model_buffer;
        if (!read_binary_from_file(option.model_filepath, &model_buffer)) {
            MD_LOG_ERROR << "Failed to read model file: " << option.model_filepath << std::endl;
            return false;
        }
        return init_from_onnx(model_buffer, option);
    }


    bool OrtBackend::init_from_onnx(const std::string& model_buffer, const RuntimeOption& option) {
        if (initialized_) {
            MD_LOG_ERROR << "OrtBackend is already initialized, cannot initialize again." << std::endl;
            return false;
        }
        try {
            build_option(option);
            session_ = {env_, model_buffer.data(), model_buffer.size(), session_options_};
            binding_ = std::make_unique<Ort::IoBinding>(session_);
            const auto input_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const auto output_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            const Ort::Allocator allocator(session_, input_memory_info);
            const size_t n_inputs = session_.GetInputCount();
            const size_t n_outputs = session_.GetOutputCount();

            // 模型输入输出信息
            tabulate::Table input_table;
            input_table.format().font_color(tabulate::Color::yellow)
                       .border_color(tabulate::Color::blue)
                       .corner_color(tabulate::Color::blue);

            // input_table.add_row(Row_t{model_info_table});
            input_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
            input_table[0].format().font_style({tabulate::FontStyle::bold});
            for (size_t i = 0; i < n_inputs; ++i) {
                auto input_name_ptr = session_.GetInputNameAllocated(i, allocator);
                auto type_info = session_.GetInputTypeInfo(i);
                std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
                const auto data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
                input_table.add_row({
                    "Input",
                    std::to_string(i),
                    input_name_ptr.get(),
                    onnx_type_to_string(data_type),
                    shape_to_string(shape)
                });
                inputs_desc_.emplace_back(TensorInfo{input_name_ptr.get(), shape, data_type});
            }

            // 输出表格
            for (size_t i = 0; i < n_outputs; ++i) {
                auto output_name_ptr = session_.GetOutputNameAllocated(i, allocator);
                auto type_info = session_.GetOutputTypeInfo(i);
                std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
                const auto data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
                input_table.add_row({
                    "Output",
                    std::to_string(i),
                    output_name_ptr.get(),
                    onnx_type_to_string(data_type),
                    shape_to_string(shape)
                });
                outputs_desc_.emplace_back(TensorInfo{output_name_ptr.get(), shape, data_type});
                binding_->BindOutput(output_name_ptr.get(), output_memory_info);
            }
            // ====================== 组合最终输出 ======================
            std::cout << termcolor::green << "[model file:"
                << std::filesystem::absolute(option.model_filepath).filename().string()
                << " model size: " << std::fixed << std::setprecision(3)
                << static_cast<float>(model_buffer.size()) / 1024 / 1024.0f << "MB]"
                << termcolor::reset << std::endl;
            std::cout << input_table << std::endl;
            initialized_ = true;
            return true;
        }
        catch (const std::exception& e) {
            MD_LOG_ERROR << "Failed to initialize from ONNX: " << e.what() << std::endl;
            return false;
        }
    }

    bool OrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs, const bool copy_to_fd) {
        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR <<
                "[OrtBackend] Size of the inputs(" << inputs.size() <<
                ") should keep same with the inputs of this model(" << inputs_desc_.size() << ")" << std::endl;
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
            MD_LOG_ERROR << "Failed to Infer: " << e.what() << std::endl;
            return false;
        }
        const std::vector<Ort::Value> ort_outputs = binding_->GetOutputValues();
        outputs->resize(ort_outputs.size());
        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            ort_value_to_md_tensor(ort_outputs[i], &(*outputs)[i], outputs_desc_[i].name);
        }
        return true;
    }

    TensorInfo OrtBackend::get_input_info(const int index) {
        return inputs_desc_[index];
    }

    std::vector<TensorInfo> OrtBackend::get_input_infos() {
        return inputs_desc_;
    }

    TensorInfo OrtBackend::get_output_info(const int index) {
        return outputs_desc_[index];
    }

    std::vector<TensorInfo> OrtBackend::get_output_infos() {
        return outputs_desc_;
    }
}

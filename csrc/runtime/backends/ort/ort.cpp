//
// Created by aichao on 2025/2/20.
//
#include <iostream>
#include <filesystem>
#include <tabulate/tabulate.hpp>
#include "csrc/runtime/backends/ort/utils.h"
#include "csrc/runtime/backends/ort/ort.h"
#include "csrc/utils/utils.h"
#include "csrc/core/md_log.h"


namespace modeldeploy {
    void OrtBackend::build_option(const OrtBackendOption& option) {
        option_ = option;
        // session_options_.SetLogSeverityLevel(1); // verbose
        //
        // session_options_.EnableProfiling(to_wstring("onnxruntime_perf_test.json").c_str());
        // std::wstring model_file = to_wstring("optimized_graph.onnx");
        // session_options_.SetOptimizedModelFilePath(model_file.c_str());

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
        const auto all_providers = Ort::GetAvailableProviders();
        std::string providers_msg;
        for (size_t i = 0; i < all_providers.size(); i++) {
            providers_msg += all_providers[i];
            if (1 != all_providers.size() - i) {
                providers_msg += ", ";
            }
        }
        // CUDA
        if (option.device == Device::GPU) {
            // 获取所有可用 Provider
            const bool has_trt = std::find(all_providers.begin(), all_providers.end(),
                                           "TensorrtExecutionProvider") != all_providers.end();
            const std::string device_id_str = std::to_string(option.device_id);
            // ==== 1. 尝试启用 TensorRT（优先） ====
            if (option.enable_trt) {
                if (has_trt) {
                    OrtTensorRTProviderOptionsV2* trt_options = nullptr;
                    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_options));

                    std::vector<const char*> keys_trt;
                    std::vector<const char*> values_trt;

                    keys_trt.push_back("device_id");
                    values_trt.push_back(device_id_str.c_str());

                    const std::string trt_fp16_enable = option.enable_fp16 ? "1" : "0";
                    keys_trt.push_back("trt_fp16_enable");
                    values_trt.push_back(trt_fp16_enable.c_str());

                    keys_trt.push_back("trt_max_workspace_size");
                    values_trt.push_back("2147483648"); // 2GB

                    keys_trt.push_back("trt_engine_cache_enable");
                    values_trt.push_back("1");

                    keys_trt.push_back("trt_engine_cache_path");
                    values_trt.push_back(option.trt_engine_cache_path.c_str());

                    // keys_trt.push_back("trt_dump_ep_context_model");
                    // values_trt.push_back("1");
                    //
                    // keys_trt.push_back("trt_ep_context_file_path");
                    // values_trt.push_back("./trt_ep_ctx");

                    // （可选）动态 shape profile 设置
                    if (!option.trt_min_shape.empty()) {
                        keys_trt.push_back("trt_profile_min_shapes");
                        values_trt.push_back(option.trt_min_shape.c_str());
                    }
                    if (!option.trt_opt_shape.empty()) {
                        keys_trt.push_back("trt_profile_opt_shapes");
                        values_trt.push_back(option.trt_opt_shape.c_str());
                    }
                    if (!option.trt_max_shape.empty()) {
                        keys_trt.push_back("trt_profile_max_shapes");
                        values_trt.push_back(option.trt_max_shape.c_str());
                    }

                    Ort::ThrowOnError(
                        Ort::GetApi().UpdateTensorRTProviderOptions(trt_options,
                                                                    keys_trt.data(), values_trt.data(),
                                                                    static_cast<int>(keys_trt.size())));
                    Ort::ThrowOnError(
                        Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options_, trt_options));
                    Ort::GetApi().ReleaseTensorRTProviderOptions(trt_options);

                    MD_LOG_INFO << "OnnxRuntime TensorrtExecutionProvider enabled." << std::endl;
                }
                else {
                    MD_LOG_WARN <<
                        "OnnxRuntime TensorrtExecutionProvider not available. "
                        "Disable TRT. Available providers: " << providers_msg << std::endl;
                }
            }

            // ==== 2. 启用 CUDA（作为 fallback 或主力） ====
            if (std::find(all_providers.begin(), all_providers.end(), "CUDAExecutionProvider") != all_providers.end()) {
                OrtCUDAProviderOptionsV2* cuda_options = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));

                std::vector<const char*> keys_cuda;
                std::vector<const char*> values_cuda;

                keys_cuda.push_back("device_id");
                values_cuda.push_back(device_id_str.c_str());

                if (option.external_stream_) {
                    const std::string stream_str = std::to_string(
                        reinterpret_cast<uintptr_t>(option.external_stream_));
                    keys_cuda.push_back("user_compute_stream");
                    values_cuda.push_back(stream_str.c_str());
                }

                Ort::ThrowOnError(
                    Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys_cuda.data(),
                                                            values_cuda.data(),
                                                            static_cast<int>(keys_cuda.size())));
                Ort::ThrowOnError(
                    Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(session_options_, cuda_options));
                Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
                MD_LOG_INFO << "OnnxRuntime CUDAExecutionProvider enabled." << std::endl;
            }
            else {
                MD_LOG_WARN << "OnnxRuntime CUDAExecutionProvider not available. Fallback to CPU. Available providers: "
                    << providers_msg << std::endl;
            }
        }
        MD_LOG_INFO << "OnnxRuntime CPUExecutionProvider enabled." << std::endl;
    }

    bool OrtBackend::init(const RuntimeOption& option) {
        if (option.device != Device::CPU && option.device != Device::GPU) {
            MD_LOG_ERROR << "Backend::ORT only supports Device::CPU/Device::GPU, but now its " << std::endl;
            return false;
        }
        option_ = option.ort_option;
        option_.device_id = option.device_id;
        option_.device = option.device;
        option_.enable_trt = option.enable_trt;
        option_.enable_fp16 = option.enable_fp16;
        option_.model_filepath = option.model_file;

        if (option_.model_from_memory) {
            return init_from_onnx(option.model_buffer, option_);
        }
        if (!std::filesystem::exists(option.model_file)) {
            MD_LOG_ERROR << "Model file does not exist: " << option.model_file << std::endl;
            return false;
        }
        if (!read_binary_from_file(option.model_file, &model_buffer_)) {
            MD_LOG_ERROR << "Failed to read model file: " << option.model_file << std::endl;
            return false;
        }
        return init_from_onnx(model_buffer_, option_);
    }


    bool OrtBackend::init_from_onnx(const std::string& model_buffer, const OrtBackendOption& option) {
        if (initialized_) {
            MD_LOG_ERROR << "OrtBackend is already initialized, cannot initialize again." << std::endl;
            return false;
        }
        try {
            build_option(option);
            shared_session_ = std::make_shared<Ort::Session>(env_, model_buffer.data(), model_buffer.size(),
                                                             session_options_);
            binding_ = std::make_unique<Ort::IoBinding>(*shared_session_);
            const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            const Ort::Allocator allocator(*shared_session_, memory_info);
            const size_t n_inputs = shared_session_->GetInputCount();
            const size_t n_outputs = shared_session_->GetOutputCount();
            // 模型输入输出信息
            tabulate::Table input_table;
            input_table.format().font_color(tabulate::Color::yellow)
                       .border_color(tabulate::Color::blue)
                       .corner_color(tabulate::Color::blue);

            // input_table.add_row(Row_t{model_info_table});
            input_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
            input_table[0].format().font_style({tabulate::FontStyle::bold});
            for (size_t i = 0; i < n_inputs; ++i) {
                auto input_name_ptr = shared_session_->GetInputNameAllocated(i, allocator);
                auto type_info = shared_session_->GetInputTypeInfo(i);
                std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
                const auto data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
                input_table.add_row({
                    "Input",
                    std::to_string(i),
                    input_name_ptr.get(),
                    onnx_type_to_string(data_type),
                    shape_to_string(shape)
                });
                inputs_desc_.emplace_back(OrtValueInfo{input_name_ptr.get(), shape, data_type});
            }

            // 输出表格
            for (size_t i = 0; i < n_outputs; ++i) {
                auto output_name_ptr = shared_session_->GetOutputNameAllocated(i, allocator);
                auto type_info = shared_session_->GetOutputTypeInfo(i);
                std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
                const auto data_type = type_info.GetTensorTypeAndShapeInfo().GetElementType();
                input_table.add_row({
                    "Output",
                    std::to_string(i),
                    output_name_ptr.get(),
                    onnx_type_to_string(data_type),
                    shape_to_string(shape)
                });
                outputs_desc_.emplace_back(OrtValueInfo{output_name_ptr.get(), shape, data_type});
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

    bool OrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR <<
                "[OrtBackend] Size of the inputs(" << inputs.size() <<
                ") should keep same with the inputs of this model(" << inputs_desc_.size() << ")" << std::endl;
            return false;
        }
        const Ort::MemoryInfo input_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto ort_value = create_ort_value(inputs[i], input_memory_info);
            binding_->BindInput(inputs_desc_[i].name.c_str(), ort_value);
        }
        const Ort::MemoryInfo output_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        for (auto& output_info : outputs_desc_) {
            binding_->BindOutput(output_info.name.c_str(), output_memory_info);
        }
        try {
            shared_session_->Run({}, *binding_);
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

    std::unique_ptr<BaseBackend> OrtBackend::clone(RuntimeOption& runtime_option,
                                                   void* stream,
                                                   const int device_id) {
        auto backend = std::make_unique<OrtBackend>();

        // 共享 Session、model_buffer、输入输出描述
        backend->shared_session_ = this->shared_session_;
        backend->model_buffer_ = this->model_buffer_;
        backend->inputs_desc_ = this->inputs_desc_;
        backend->outputs_desc_ = this->outputs_desc_;
        backend->option_ = this->option_;

        runtime_option.device = (device_id >= 0) ? Device::GPU : Device::CPU;
        runtime_option.device_id = device_id;
        runtime_option.ort_option.device = runtime_option.device;
        runtime_option.ort_option.device_id = device_id;

        if (stream) {
            runtime_option.ort_option.external_stream_ = stream;
        }
        // 每个线程各自持有自己的 binding
        backend->binding_ = std::make_unique<Ort::IoBinding>(*backend->shared_session_);
        return backend;
    }


    TensorInfo OrtBackend::get_input_info(const int index) {
        TensorInfo info;
        info.name = inputs_desc_[index].name;
        info.shape.assign(inputs_desc_[index].shape.begin(),
                          inputs_desc_[index].shape.end());
        info.dtype = get_md_dtype(inputs_desc_[index].dtype);
        return info;
    }

    std::vector<TensorInfo> OrtBackend::get_input_infos() {
        auto size = inputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < size; i++) {
            infos.emplace_back(get_input_info(i));
        }
        return infos;
    }

    TensorInfo OrtBackend::get_output_info(const int index) {
        TensorInfo info;
        info.name = outputs_desc_[index].name;
        info.shape.assign(outputs_desc_[index].shape.begin(),
                          outputs_desc_[index].shape.end());
        info.dtype = get_md_dtype(outputs_desc_[index].dtype);
        return info;
    }

    std::vector<TensorInfo> OrtBackend::get_output_infos() {
        auto size = outputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < outputs_desc_.size(); i++) {
            infos.emplace_back(get_output_info(i));
        }
        return infos;
    }

    std::map<std::string, std::string> OrtBackend::get_custom_meta_data() const {
        std::map<std::string, std::string> data;
        const Ort::AllocatorWithDefaultOptions allocator;
        const Ort::ModelMetadata model_metadata = shared_session_->GetModelMetadata();
        // 获取自定义元数据数量
        const auto keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
        // 遍历所有自定义元数据
        for (const auto& key_ptr : keys) {
            const char* key = key_ptr.get();
            auto value = model_metadata.LookupCustomMetadataMapAllocated(key, allocator);
            data[std::string(key)] = std::string(value.get());
        }
        return data;
    }
}

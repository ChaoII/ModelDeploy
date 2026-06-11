//
// Created by aichao on 2025/6/27.
//
#include "runtime/backends/trt/trt_backend.h"

#include <cstring>
#include <fstream>
#include <tabulate/tabulate.hpp>
#include "NvInferRuntime.h"
#include "utils/utils.h"
#include "runtime/backends/trt/buffers.h"


namespace modeldeploy {
    MDTrtLogger* MDTrtLogger::logger = nullptr;


    bool TrtBackend::init(const RuntimeOption& option) {
        if (option.device != Device::GPU) {
            MD_LOG_ERROR << "TrtBackend only supports Device::GPU, but now it's "
                << option.device << "." << std::endl;
            return false;
        }
        option_ = option.trt_option;
        option_.model_file = option.model_file;
        option_.gpu_id = option.device_id;
        option_.enable_fp16 = option.enable_fp16; // 同步用户设置的 FP16 标志
        const cudaError_t error = cudaSetDevice(option_.gpu_id);
        if (error != cudaSuccess) {
            MD_LOG_ERROR << "Failed to set CUDA device: " << error << std::endl;
            return false;
        }
        if (option.model_from_memory || option_.model_from_memory) {
            runtime_.reset(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
            auto& buf = option.model_from_memory ? option.model_buffer : option_.model_buffer;
            return load_trt_cache(buf);
        }
        if (!std::filesystem::exists(option.model_file)) {
            MD_LOG_ERROR << "Model file does not exist: " << option.model_file << std::endl;
            return false;
        }

        // Check if this is an ONNX model (needs build) or pre-built engine
        const bool is_onnx = option.model_file.size() > 5 &&
            (option.model_file.substr(option.model_file.size() - 5) == ".onnx");

        if (is_onnx) {
            // --- Build TRT engine from ONNX ---
            // Auto-generate cache path: model.onnx → model.onnx.engine
            auto cache_path = option_.cache_file_path;
            if (cache_path.empty()) {
                cache_path = option.model_file + ".engine";
            }

            // Check cache first
            if (std::filesystem::exists(cache_path)) {
                auto onnx_time = std::filesystem::last_write_time(option.model_file);
                auto cache_time = std::filesystem::last_write_time(cache_path);
                if (cache_time >= onnx_time) {
                    MD_LOG_INFO << "Loading cached TRT engine: " << cache_path << std::endl;
                    runtime_.reset(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
                    if (!read_binary_from_file(cache_path, &model_buffer_)) {
                        MD_LOG_ERROR << "Failed to read cached engine file." << std::endl;
                        return false;
                    }
                    return load_trt_cache(model_buffer_);
                }
            }

            // Build engine from ONNX
            MD_LOG_INFO << "Building TensorRT engine from ONNX: " << option.model_file << std::endl;
            auto builder = std::unique_ptr<nvinfer1::IBuilder>(
                nvinfer1::createInferBuilder(*MDTrtLogger::get()));
            if (!builder) { MD_LOG_ERROR << "Failed to create TRT builder." << std::endl; return false; }

            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
                builder->createNetworkV2(explicitBatch));
            if (!network) { MD_LOG_ERROR << "Failed to create TRT network." << std::endl; return false; }

            auto parser = std::unique_ptr<nvonnxparser::IParser>(
                nvonnxparser::createParser(*network, *MDTrtLogger::get()));
            if (!parser) { MD_LOG_ERROR << "Failed to create ONNX parser." << std::endl; return false; }

            if (!parser->parseFromFile(option.model_file.c_str(),
                                       static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
                MD_LOG_ERROR << "Failed to parse ONNX model." << std::endl;
                return false;
            }

            auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
            if (!config) { MD_LOG_ERROR << "Failed to create TRT config." << std::endl; return false; }

            // === 与 trtexec 一致的优化配置 ===
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, option_.max_workspace_size);
            config->setBuilderOptimizationLevel(5); // 最高优化等级（trtexec 默认 3）

            if (option_.enable_fp16) {
                if (builder->platformHasFastFp16()) {
                    config->setFlag(nvinfer1::BuilderFlag::kFP16);
                } else {
                    MD_LOG_WARN << "FP16 not supported on this platform, falling back to FP32." << std::endl;
                }
            }

            // 启用 TF32（Ampere+ 架构默认可用，trtexec 默认启用）
            if (builder->platformHasTf32()) {
                config->setFlag(nvinfer1::BuilderFlag::kTF32);
            }

            // Timing cache：让 builder 通过实际 kernel timing 选择最快实现
            // 否则 builder 用 heuristic 选 kernel，性能差 1-3ms
            auto timing_cache = std::unique_ptr<nvinfer1::ITimingCache>(
                config->createTimingCache(nullptr, 0));
            if (timing_cache) {
                config->setTimingCache(*timing_cache, "global");
            }

            // Auto-detect dynamic shapes and set optimization profiles
            bool has_dynamic = false;
            if (option_.min_shape.empty()) {
                // No user-provided profile — check network for dynamic dims
                auto profile = builder->createOptimizationProfile();
                for (int i = 0; i < network->getNbInputs(); ++i) {
                    auto* input = network->getInput(i);
                    auto dims = input->getDimensions();
                    if (dims.nbDims < 0) continue;
                    bool input_dynamic = false;

                    // Detect the layout: for 4D [N,C,H,W] image inputs,
                    // infer a reasonable opt size from static dims (e.g. C=3)
                    bool is_image_4d = (dims.nbDims == 4);
                    for (int d = 0; d < dims.nbDims; ++d) {
                        if (dims.d[d] == -1) {
                            input_dynamic = true;
                            has_dynamic = true;
                        }
                    }
                    if (!input_dynamic) continue;

                    nvinfer1::Dims min_dims = dims, opt_dims = dims, max_dims = dims;
                    for (int d = 0; d < dims.nbDims; ++d) {
                        if (dims.d[d] != -1) {
                            // Static dimension — use as-is
                            min_dims.d[d] = dims.d[d];
                            opt_dims.d[d] = dims.d[d];
                            max_dims.d[d] = dims.d[d];
                            continue;
                        }
                        // Dynamic dimension — infer reasonable range
                        if (d == 0) {
                            // Batch dimension
                            min_dims.d[d] = 1;
                            opt_dims.d[d] = 1;
                            max_dims.d[d] = 4;
                        }
                        else if (is_image_4d && (d == 2 || d == 3)) {
                            // Spatial H/W for 4D image input
                            min_dims.d[d] = 32;
                            opt_dims.d[d] = 640; // common default; overridable via MDRuntimeOption
                            max_dims.d[d] = 1280;
                        }
                        else {
                            // Other dynamic dimensions (e.g. sequence length)
                            min_dims.d[d] = 1;
                            opt_dims.d[d] = 64;
                            max_dims.d[d] = 512;
                        }
                    }
                    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
                    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
                    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
                    auto dims_to_str = [](const nvinfer1::Dims& d) {
                        std::string s = "[";
                        for (int j = 0; j < d.nbDims; ++j) { if (j) s += ","; s += std::to_string(d.d[j]); }
                        return s + "]";
                    };
                    MD_LOG_INFO << "TRT auto profile [" << input->getName() << "]: "
                        << "min=" << dims_to_str(min_dims) << " "
                        << "opt=" << dims_to_str(opt_dims) << " "
                        << "max=" << dims_to_str(max_dims)
                        << " (override via MDRuntimeOption.trt_min_shape)" << std::endl;
                }
                if (has_dynamic) {
                    config->addOptimizationProfile(profile);
                }
            }
            else {
                // User-provided profiles
                auto profile = builder->createOptimizationProfile();
                for (auto& [name, min_dims] : option_.min_shape) {
                    auto opt_it = option_.opt_shape.find(name);
                    auto max_it = option_.max_shape.find(name);
                    const auto& opt_dims = (opt_it != option_.opt_shape.end()) ? opt_it->second : min_dims;
                    const auto& max_dims = (max_it != option_.max_shape.end()) ? max_it->second : min_dims;
                    profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, vec_to_dims(min_dims));
                    profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, vec_to_dims(opt_dims));
                    profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, vec_to_dims(max_dims));
                }
                config->addOptimizationProfile(profile);
            }

            auto* serialized_engine = builder->buildSerializedNetwork(*network, *config);
            if (!serialized_engine) {
                MD_LOG_ERROR << "Failed to build TRT engine from ONNX: " << option.model_file
                    << ". The ONNX model may contain ops unsupported by TensorRT (e.g. NMS). "
                    << "Try using the model without embedded NMS (e.g. yolo11n.onnx instead of yolo11n_nms.onnx), "
                    << "or use ORT backend (MD_BACKEND_ORT) with option.enable_trt=1 for mixed EP execution."
                    << std::endl;
                return false;
            }

            // Save cache
            {
                std::ofstream ofs(cache_path, std::ios::binary);
                if (ofs) {
                    ofs.write(static_cast<const char*>(serialized_engine->data()),
                              static_cast<std::streamsize>(serialized_engine->size()));
                    MD_LOG_INFO << "TRT engine cached to: " << cache_path << std::endl;
                }
            }

            // Deserialize into runtime engine
            runtime_.reset(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
            model_buffer_.assign(static_cast<const char*>(serialized_engine->data()),
                                 serialized_engine->size());
            delete serialized_engine;
            if (cudaStreamCreate(&stream_) != 0) {
                MD_LOG_FATAL << "Cannot call cudaStreamCreate()." << std::endl;
            }
            return load_trt_cache(model_buffer_);
        }
        else {
            // --- Load pre-built .engine file ---
            runtime_.reset(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
            if (!read_binary_from_file(option.model_file, &model_buffer_)) {
                MD_LOG_ERROR << "Failed to read model file: " << option.model_file << std::endl;
                return false;
            }
            MD_LOG_INFO << "[TensorRT engine loaded, size: "
                << std::fixed << std::setprecision(3)
                << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]" << std::endl;
            if (cudaStreamCreate(&stream_) != 0) {
                MD_LOG_FATAL << "Cannot call cudaStreamCreate()." << std::endl;
            }
            return load_trt_cache(model_buffer_);
        }
    }

    TrtBackend::~TrtBackend() {
        // 重要：必须按照正确的顺序销毁对象
        if (stream_ != nullptr && option_.external_stream != nullptr) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        // 1. 首先销毁执行上下文
        context_.reset();
        // 2. 销毁引擎（必须在运行时之前销毁）
        engine_.reset();
        // 3. 最后销毁运行时
        runtime_.reset();
    }

    bool TrtBackend::load_trt_cache(const std::string& engine_buffer) {
        engine_.reset(runtime_->
                      deserializeCudaEngine(engine_buffer.data(),
                                            engine_buffer.size()), InferDeleter());
        if (!engine_) {
            MD_LOG_ERROR << "Deserialize engine failed";
            return false;
        }
        // 4. 打印输入输出信息
        tabulate::Table io_table;
        io_table.format().font_color(tabulate::Color::yellow)
                .border_color(tabulate::Color::blue)
                .corner_color(tabulate::Color::blue);
        io_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
        io_table[0].format().font_style({tabulate::FontStyle::bold});
        context_.reset(engine_->createExecutionContext());
        for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
            auto const name = engine_->getIOTensorName(i);
            const bool is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            const auto dims = engine_->getTensorShape(name);
            const auto dtype = engine_->getTensorDataType(name);
            if (is_input) {
                inputs_desc_.emplace_back(TrtValueInfo{name, dims_to_vec(dims), dtype});
            }
            else {
                outputs_desc_.emplace_back(TrtValueInfo{name, dims_to_vec(dims), dtype});
            }
            std::string shape_str = vector_to_string(dims_to_vec(dims));
            std::string type_str = datatype_to_string(trt_dtype_to_md_dtype(dtype));
            io_table.add_row({
                is_input ? "Input" : "Output",
                std::to_string(i), name, type_str, shape_str
            });
        }
        MD_LOG_INFO << std::endl << io_table << std::endl;
        // Log optimization profile ranges
        const int nb_profiles = engine_->getNbOptimizationProfiles();
        if (nb_profiles > 0) {
            for (int p = 0; p < nb_profiles; ++p) {
                for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
                    auto const name = engine_->getIOTensorName(i);
                    if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
                    auto min_d = engine_->getProfileShape(name, p, nvinfer1::OptProfileSelector::kMIN);
                    auto opt_d = engine_->getProfileShape(name, p, nvinfer1::OptProfileSelector::kOPT);
                    auto max_d = engine_->getProfileShape(name, p, nvinfer1::OptProfileSelector::kMAX);
                    MD_LOG_INFO << "TRT profile[" << p << "] " << name
                        << ": min=" << vector_to_string(dims_to_vec(min_d))
                        << " opt=" << vector_to_string(dims_to_vec(opt_d))
                        << " max=" << vector_to_string(dims_to_vec(max_d)) << std::endl;
                }
            }
        }
        return true;
    }

    bool TrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (inputs.size() != num_inputs()) {
            MD_LOG_ERROR << "Require " << num_inputs() << " inputs, but got " << inputs.size() << "." << std::endl;
            return false;
        }

        // 推理期间保持所有 device buffer 存活
        std::vector<CudaBufferPrt> device_buffers;
        last_input_shapes_.resize(inputs.size());

        // ---- 处理输入 ----
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto& input = inputs[j];
            const std::string& name = input.get_name();
            nvinfer1::Dims dims = vec_to_dims(input.shape());

            // 仅当 shape 变化时才调 setInputShape（避免不必要的 validation 开销）
            if (j >= last_input_shapes_.size() ||
                std::memcmp(last_input_shapes_[j].d, dims.d, dims.nbDims * sizeof(int32_t)) != 0) {
                if (!context_->setInputShape(name.c_str(), dims)) {
                    MD_LOG_ERROR << "Failed to set input shape for " << name
                        << " [" << vector_to_string(input.shape()) << "]. "
                        << "The shape is outside the engine's optimization profile range. "
                        << "Delete the cached .engine file to rebuild from ONNX."
                        << std::endl;
                    return false;
                }
                last_input_shapes_[j] = dims;
            }

            const size_t buffer_size = input.byte_size();
            CudaBufferPrt input_buffer = nullptr;
            if (input.device() == Device::GPU) {
                input_buffer = std::make_unique<CudaBuffer>();
                input_buffer->shared_from_external(buffer_size, input.data());
            }
            else {
                input_buffer = allocate_cuda_buffer(buffer_size);
                input_buffer->copy_from_host(input.data());
            }
            context_->setTensorAddress(name.c_str(), input_buffer->data());
            device_buffers.push_back(std::move(input_buffer));
        }

        // ---- 处理输出（复用 buffer） ----
        cached_output_buffers_.resize(outputs_desc_.size());
        for (size_t j = 0; j < outputs_desc_.size(); j++) {
            auto name = outputs_desc_[j].name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            const size_t buffer_size = volume(dims) * trt_data_type_size(outputs_desc_[j].dtype);

            // 只在 buffer 不够大时重新分配
            if (!cached_output_buffers_[j] || cached_output_buffers_[j]->byte_size() < buffer_size) {
                cached_output_buffers_[j] = allocate_cuda_buffer(buffer_size);
            }
            context_->setTensorAddress(name.c_str(), cached_output_buffers_[j]->data());
        }

        // ---- 执行推理 ----
        if (!context_->enqueueV3(stream_)) {
            MD_LOG_ERROR << "Failed to run inference with TensorRT." << std::endl;
            return false;
        }

        // ---- 异步拷贝输出到主机（无需中间的 cudaStreamSynchronize） ----
        outputs->resize(outputs_desc_.size());
        for (size_t j = 0; j < outputs_desc_.size(); j++) {
            auto name = outputs_desc_[j].name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
            (*outputs)[j].allocate(shape, DataType::FP32, Device::CPU, name);
            const void* device_buffer = context_->getTensorAddress(name.c_str());
            cudaMemcpyAsync((*outputs)[j].data(), device_buffer, (*outputs)[j].byte_size(),
                            cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);
        return true;
    }


    TensorInfo TrtBackend::get_input_info(int index) {
        if (index >= num_inputs()) {
            MD_LOG_FATAL << "The index: " << index << " should less than the number of inputs: "
                << num_inputs() << "." << std::endl;
        }

        TensorInfo info;
        info.name = inputs_desc_[index].name;
        info.shape = inputs_desc_[index].shape;
        info.dtype = trt_dtype_to_md_dtype(inputs_desc_[index].dtype);
        return info;
    }

    std::vector<TensorInfo> TrtBackend::get_input_infos() {
        const auto size = inputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < size; i++) {
            infos.emplace_back(get_input_info(i));
        }
        return infos;
    }

    TensorInfo TrtBackend::get_output_info(const int index) {
        if (index >= num_outputs()) {
            MD_LOG_FATAL << "The index: " << index << " should less than the number of outputs: "
                << num_outputs() << "." << std::endl;
        }
        TensorInfo info;
        info.name = outputs_desc_[index].name;
        info.shape.assign(outputs_desc_[index].shape.begin(),
                          outputs_desc_[index].shape.end());
        info.dtype = trt_dtype_to_md_dtype(outputs_desc_[index].dtype);
        return info;
    }

    std::vector<TensorInfo> TrtBackend::get_output_infos() {
        const auto size = outputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < size; i++) {
            infos.emplace_back(get_output_info(i));
        }
        return infos;
    }

    std::unique_ptr<BaseBackend> TrtBackend::clone(const RuntimeOption& runtime_option,
                                                   void* stream, const int device_id) {
        auto new_backend = std::make_unique<TrtBackend>();
        if (device_id > 0 && device_id != option_.gpu_id) {
            auto clone_option = option_;
            clone_option.gpu_id = device_id;
            clone_option.external_stream = stream;
            std::string model_buffer;
            if (!read_binary_from_file(runtime_option.model_file, &model_buffer)) {
                MD_LOG_FATAL << "Fail to read binary from model file while cloning TrtBackend" << std::endl;
            }
            if (!new_backend->load_trt_cache(model_buffer)) {
                MD_LOG_FATAL << "Clone model from engine file initialize TrtBackend." << std::endl;
            }
            return new_backend;
        }
        new_backend->option_.gpu_id = option_.gpu_id;
        if (stream) {
            new_backend->stream_ = static_cast<cudaStream_t>(stream);
        }
        else {
            if (cudaSetDevice(new_backend->option_.gpu_id) != cudaSuccess) {
                MD_LOG_ERROR << "Cant not call cudaSetDevice()." << std::endl;
                return nullptr;
            }
            if (cudaStreamCreate(&new_backend->stream_) != cudaSuccess) {
                MD_LOG_ERROR << "Cant not call cudaStreamCreate()." << std::endl;
                return nullptr;
            }
        }

        new_backend->engine_ = engine_;
        new_backend->context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            new_backend->engine_->createExecutionContext());
        if (!new_backend->context_) {
            MD_LOG_ERROR << "Failed to create execution context for clone." << std::endl;
            return nullptr;
        }
        new_backend->inputs_desc_ = inputs_desc_;
        new_backend->outputs_desc_ = outputs_desc_;
        return new_backend;
    }
} // namespace modeldeploy

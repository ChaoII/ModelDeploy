//
// Created by aichao on 2025/6/27.
//
#include "runtime/backends/trt/trt_backend_impl.h"

#include <cstring>
#include <fstream>
#include <tabulate/tabulate.hpp>
#include "NvInferRuntime.h"
#include "utils/utils.h"
#include "runtime/backends/trt/buffers.h"


namespace modeldeploy {
    MDTrtLogger* MDTrtLogger::logger = nullptr;

    bool CanBuildEngine(
        const std::map<std::string, ShapeRangeInfo>& shape_range_info) {
        for (auto iter = shape_range_info.begin(); iter != shape_range_info.end(); ++iter) {
            bool is_full_static = true;
            for (size_t i = 0; i < iter->second.shape.size(); ++i) {
                if (iter->second.shape[i] < 0) {
                    is_full_static = false;
                    break;
                }
            }

            if (is_full_static) {
                continue;
            }
            for (size_t i = 0; i < iter->second.shape.size(); ++i) {
                if (iter->second.min[i] < 0 || iter->second.max[i] < 0) {
                    return false;
                }
            }
        }
        return true;
    }

    bool TrtBackendImpl::load_trt_cache(const std::string& engine_buffer) {
        cudaSetDevice(option_.gpu_id);

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(*MDTrtLogger::get())
        );
        if (!runtime_) {
            MD_LOG_ERROR << "Failed to call createInferRuntime()." << std::endl;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()),
            InferDeleter());
        if (!engine_) {
            MD_LOG_ERROR << "Failed to call deserializeCudaEngine()." << std::endl;
            return false;
        }
        get_input_output_info();
        context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());

        for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
            auto const name = engine_->getIOTensorName(i);

            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }
            auto min = dims_to_vec(engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX));
            auto max = dims_to_vec(engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMIN));
        }
        MD_LOG_INFO << "Build TensorRT Engine from cache"
            << " with shape range information as below," << std::endl;
        for (const auto& item : shape_range_info_) {
            MD_LOG_INFO << item.second << std::endl;
        }
        return true;
    }

    bool TrtBackendImpl::init(const RuntimeOption& option) {
        auto trt_option = option.trt_option;
        trt_option.model_file = option.model_file;
        trt_option.gpu_id = option.device_id;
        option_ = trt_option;
        if (option.device != Device::GPU) {
            MD_LOG_ERROR << "TrtBackend only supports Device::GPU, but now it's "
                << option.device << "." << std::endl;
            return false;
        }

        if (option.model_from_memory) {
            return load_trt_cache(option.model_buffer);
        }
        if (!std::filesystem::exists(option.model_file)) {
            MD_LOG_ERROR << "Model file does not exist: " << option.model_file << std::endl;
            return false;
        }
        if (!read_binary_from_file(option.model_file, &model_buffer_)) {
            MD_LOG_ERROR << "Failed to read model file: " << option.model_file << std::endl;
            return false;
        }
        constexpr uint32_t flags =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

        builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*MDTrtLogger::get()));
        if (!builder_) {
            MD_LOG_ERROR << "Failed to call createInferBuilder()." << std::endl;
            return false;
        }
        network_ = std::unique_ptr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(flags));
        if (!network_) {
            MD_LOG_ERROR << "Failed to call createNetworkV2()." << std::endl;
            return false;
        }
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
        std::cout << termcolor::green << "[TensorRT model buffer loaded, size: "
            << std::fixed << std::setprecision(3)
            << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]"
            << termcolor::reset << std::endl;
        // return init_from_onnx(model_buffer_, option_);
        if (cudaStreamCreate(&stream_) != 0) {
            MD_LOG_FATAL << "Cant not call cudaStreamCreate()." << std::endl;
        }
        // if (!trt_option.cache_file_path.empty())
        //     return load_trt_cache(trt_option.cache_file_path);
        return load_trt_cache(model_buffer_);
    }


    bool TrtBackendImpl::init_from_onnx(const std::string& model_buffer) {
        if (initialized_) {
            std::cerr << "TrtBackend is already initialized, cannot initialize again." << std::endl;
            return false;
        }

        // 1. 创建 ONNX Parser
        auto parser = nvonnxparser::createParser(*network_, *MDTrtLogger::get());
        if (!parser->parse(model_buffer.c_str(), model_buffer.size())) {
            std::cerr << "Failed to parse ONNX model with TensorRT" << std::endl;
            return false;
        }
        // 2. 构建 engine
        const auto config = builder_->createBuilderConfig();
        const auto profile = builder_->createOptimizationProfile();
        for (const auto& item : shape_range_info_) {
            if (!profile->setDimensions(item.first.c_str(),
                                        nvinfer1::OptProfileSelector::kMIN,
                                        vec_to_dims(item.second.min))) {
                MD_LOG_FATAL << "[TrtBackend] Failed to set min_shape for input: " << item.first <<
                    " in TrtBackend." << std::endl;
            }

            if (!profile->setDimensions(item.first.c_str(),
                                        nvinfer1::OptProfileSelector::kMAX, vec_to_dims(item.second.max))) {
                MD_LOG_FATAL << "[TrtBackend] Failed to set max_shape for input: " << item.first <<
                    " in TrtBackend." << std::endl;
            }
            if (item.second.opt.empty()) {
                if (!profile->setDimensions(item.first.c_str(),
                                            nvinfer1::OptProfileSelector::kOPT, vec_to_dims(item.second.max))) {
                    MD_LOG_FATAL << "[TrtBackend] Failed to set opt_shape for input: " << item.first <<
                        " in TrtBackend." << std::endl;
                }
            }
            else {
                if (item.second.opt.size() != item.second.shape.size()) {
                    MD_LOG_FATAL << "Require the dimension of opt in shape range information equal to "
                        "dimension of input: " << item.first << " in this model, but now it's " << item.second.opt.
                        size() << " != " << item.second.shape.size() << "." << std::endl;
                }

                if (!
                    profile->setDimensions(item.first.c_str(),
                                           nvinfer1::OptProfileSelector::kOPT,
                                           vec_to_dims(item.second.opt))) {
                    MD_LOG_FATAL << "[TrtBackend] Failed to set opt_shape for input: " << item.first <<
                        " in TrtBackend." << std::endl;
                }
            }
        }
        config->addOptimizationProfile(profile);
        std::unique_ptr<nvinfer1::IHostMemory> plan = std::unique_ptr<nvinfer1::IHostMemory>{
            builder_->buildSerializedNetwork(*network_, *config)
        };
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(plan->data(), plan->size()));

        if (!engine_) {
            std::cerr << "Failed to build TensorRT engine." << std::endl;
            return false;
        }

        // 3. 创建上下文
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Failed to create execution context." << std::endl;
            return false;
        }
        initialized_ = true;
        return true;
    }

    int TrtBackendImpl::shape_range_info_updated(const std::vector<Tensor>& inputs) {
        bool need_update_engine = false;
        for (const auto& input : inputs) {
            auto iter = shape_range_info_.find(input.get_name());
            if (iter == shape_range_info_.end()) {
                MD_LOG_ERROR << "There's no input named '" << input.get_name() << "' in loaded model." << std::endl;
            }
            if (iter->second.update(input.shape()) == 1) {
                need_update_engine = true;
            }
        }
        return need_update_engine;
    }

    bool TrtBackendImpl::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (inputs.size() != num_inputs()) {
            MD_LOG_ERROR << "Require " << num_inputs() << " inputs, but got " << inputs.size() << "." << std::endl;
            return false;
        }

        cudaError_t error = cudaSetDevice(option_.gpu_id);
        if (error != cudaSuccess) {
            MD_LOG_ERROR << "Failed to set CUDA device: " << error << std::endl;
            return false;
        }
        std::vector<CudaBufferPrt> device_input_buffers;
        std::vector<CudaBufferPrt> device_output_buffers;
        // 处理输入缓冲区
        for (auto& input : inputs) {
            const std::string& name = input.get_name();
            nvinfer1::Dims dims = vec_to_dims(input.shape());
            // 设置输入形状
            if (!context_->setInputShape(name.c_str(), dims)) {
                MD_LOG_ERROR << "Failed to set input shape for " << name;
                return false;
            }
            // 分配并复制数据到设备
            const size_t buffer_size = input.byte_size();
            device_input_buffers.emplace_back(allocate_cuda_buffer(buffer_size));
            device_input_buffers.back()->copy_to_device(input.data());
        }

        // 处理输出缓冲区
        for (auto& output_desc : outputs_desc_) {
            auto name = output_desc.name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            size_t buffer_size = volume(dims) * trt_data_type_size(output_desc.dtype);
            // 分配输出缓冲区
            device_output_buffers.emplace_back(allocate_cuda_buffer(buffer_size));
        }

        // 准备TensorRT的输入输出绑定
        std::vector<void*> bindings;
        bindings.reserve(inputs.size() + outputs_desc_.size());
        // 逐个元素添加进 bin
        for (const auto& buffer : device_input_buffers) {
            bindings.push_back(buffer->data());
        }
        // 输出绑定
        for (const auto& buffer : device_output_buffers) {
            bindings.push_back(buffer->data());
        }
        // 执行推理
        if (!context_->executeV2(bindings.data())) {
            MD_LOG_ERROR << "Failed to run inference with TensorRT." << std::endl;
            return false;
        }
        // 复制输出数据回主机
        outputs->resize(outputs_desc_.size());
        for (int i = 0; i < outputs_desc_.size(); i++) {
            auto name = outputs_desc_[i].name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
            (*outputs)[i].allocate(shape, DataType::FP32, name);
            device_output_buffers[i]->copy_to_host((*outputs)[i].data());
        }
        return true;
    }


    void TrtBackendImpl::get_input_output_info() {
        // 4. 打印输入输出信息
        tabulate::Table io_table;
        io_table.format().font_color(tabulate::Color::yellow)
                .border_color(tabulate::Color::blue)
                .corner_color(tabulate::Color::blue);
        io_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
        io_table[0].format().font_style({tabulate::FontStyle::bold});

        const int nb_bindings = engine_->getNbIOTensors();
        for (int i = 0; i < nb_bindings; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const nvinfer1::Dims dims = engine_->getTensorShape(name);
            const nvinfer1::DataType dtype = engine_->getTensorDataType(name);
            const bool is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            std::string shape_str = vector_to_string(dims_to_vec(dims));
            std::string type_str = datatype_to_string(trt_dtype_to_md_dtype(dtype));
            io_table.add_row({
                is_input ? "Input" : "Output",
                std::to_string(i),
                name,
                type_str,
                shape_str
            });
            if (is_input) {
                inputs_desc_.emplace_back(TrtValueInfo{name, dims_to_vec(dims), dtype});
            }
            else {
                outputs_desc_.emplace_back(TrtValueInfo{name, dims_to_vec(dims), dtype});
            }
        }
        std::cout << io_table << std::endl;
    }


    TensorInfo TrtBackendImpl::get_input_info(int index) {
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

    std::vector<TensorInfo> TrtBackendImpl::get_input_infos() {
        const auto size = inputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < size; i++) {
            infos.emplace_back(get_input_info(i));
        }
        return infos;
    }

    TensorInfo TrtBackendImpl::get_output_info(int index) {
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

    std::vector<TensorInfo> TrtBackendImpl::get_output_infos() {
        const auto size = outputs_desc_.size();
        std::vector<TensorInfo> infos;
        infos.reserve(size);
        for (auto i = 0; i < size; i++) {
            infos.emplace_back(get_output_info(i));
        }
        return infos;
    }

    std::unique_ptr<TrtBackendImpl> TrtBackendImpl::clone(RuntimeOption& runtime_option,
                                                          void* stream, int device_id) {
        std::unique_ptr<TrtBackendImpl> new_backend = std::make_unique<TrtBackendImpl>();
        auto casted_backend = dynamic_cast<TrtBackendImpl*>(new_backend.get());
        if (device_id > 0 && device_id != option_.gpu_id) {
            auto clone_option = option_;
            clone_option.gpu_id = device_id;
            clone_option.external_stream_ = stream;
            std::string model_buffer;
            if (!
                read_binary_from_file(clone_option.model_file, &model_buffer)) {
                MD_LOG_FATAL << "Fail to read binary from model file while cloning TrtBackend" << std::endl;
            }
            if (!casted_backend->init_from_onnx(model_buffer)) {
                MD_LOG_FATAL << "Clone model from ONNX failed while initialize TrtBackend." << std::endl;
            }

            return new_backend;
        }
        cudaSetDevice(option_.gpu_id);
        casted_backend->option_.gpu_id = option_.gpu_id;
        if (stream) {
            casted_backend->stream_ = reinterpret_cast<cudaStream_t>(stream);
        }
        else {
            if (cudaStreamCreate(&casted_backend->stream_) != 0) {
                MD_LOG_FATAL << "Cant not call cudaStreamCreate()." << std::endl;;
            }
        }
        casted_backend->inputs_desc_.assign(inputs_desc_.begin(), inputs_desc_.end());
        casted_backend->outputs_desc_.assign(outputs_desc_.begin(),
                                             outputs_desc_.end());

        casted_backend->shape_range_info_.insert(shape_range_info_.begin(), shape_range_info_.end());
        casted_backend->engine_ = engine_;
        casted_backend->context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            casted_backend->engine_->createExecutionContext());
        casted_backend->get_input_output_info();
        MD_LOG_INFO << "TRTBackend clone finish." << std::endl;
        return new_backend;
    }
} // namespace modeldeploy

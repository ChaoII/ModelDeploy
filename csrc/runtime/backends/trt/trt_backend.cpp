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
        cudaSetDevice(option_.gpu_id);
        runtime_.reset(nvinfer1::createInferRuntime(*MDTrtLogger::get()));
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
        std::cout << termcolor::green << "[TensorRT model buffer loaded, size: "
            << std::fixed << std::setprecision(3)
            << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]"
            << termcolor::reset << std::endl;
        if (cudaStreamCreate(&stream_) != 0) {
            MD_LOG_FATAL << "Cant not call cudaStreamCreate()." << std::endl;
        }
        return load_trt_cache(model_buffer_);
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
        std::cout << io_table << std::endl;
        return true;
    }

    bool TrtBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (inputs.size() != num_inputs()) {
            MD_LOG_ERROR << "Require " << num_inputs() << " inputs, but got " << inputs.size() << "." << std::endl;
            return false;
        }
        const cudaError_t error = cudaSetDevice(option_.gpu_id);
        if (error != cudaSuccess) {
            MD_LOG_ERROR << "Failed to set CUDA device: " << error << std::endl;
            return false;
        }
        // z自动管理分贝的cuda 内存
        std::vector<CudaBufferPrt> device_buffers;
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
            auto input_buffer = allocate_cuda_buffer(buffer_size);
            input_buffer->copy_from_host(input.data());
            context_->setTensorAddress(name.c_str(), input_buffer->data());
            device_buffers.push_back(std::move(input_buffer));
        }

        // 处理输出缓冲区
        for (auto& output_desc : outputs_desc_) {
            auto name = output_desc.name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            const size_t buffer_size = volume(dims) * trt_data_type_size(output_desc.dtype);
            // 分配输出缓冲区
            auto output_buffer = allocate_cuda_buffer(buffer_size);
            context_->setTensorAddress(name.c_str(), output_buffer->data());
            device_buffers.push_back(std::move(output_buffer));
        }

        // 执行推理
        if (!context_->enqueueV3(stream_)) {
            MD_LOG_ERROR << "Failed to run inference with TensorRT." << std::endl;
            return false;
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            MD_LOG_ERROR << "CUDA error after inference, cudaStreamSynchronize" << std::endl;
            return false;
        }
        // 复制输出数据回主机
        outputs->resize(outputs_desc_.size());
        for (int i = 0; i < outputs_desc_.size(); i++) {
            auto name = outputs_desc_[i].name;
            nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
            std::vector shape(dims.d, dims.d + dims.nbDims);
            (*outputs)[i].allocate(shape, DataType::FP32, name);
            const void* device_buffer = context_->getTensorAddress(name.c_str());
            cudaMemcpyAsync((*outputs)[i].data(), device_buffer, (*outputs)[i].byte_size(),
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

    std::unique_ptr<TrtBackend> TrtBackend::clone(const RuntimeOption& runtime_option,
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

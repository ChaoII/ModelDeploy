// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/runtime/backends/trt/trt.h"

#include <cstring>
#include <fstream>
#include <unordered_map>

#include "NvInferRuntime.h"
#include "csrc/utils/utils.h"


namespace modeldeploy {
    FDTrtLogger* FDTrtLogger::logger = nullptr;

    // Check if the model can build tensorrt engine now
    // If the model has dynamic input shape, it will require defined shape
    // information We can set the shape range information by function
    // SetTrtInputShape() But if the shape range is not defined, then the engine
    // cannot build, in this case, The engine will build once there's data feeded,
    // and the shape range will be updated
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

    bool TrtBackend::LoadTrtCache(const std::string& trt_engine_file) {
        cudaSetDevice(option_.gpu_id);

        std::string engine_buffer;
        if (!read_binary_from_file(trt_engine_file, &engine_buffer)) {
            MD_LOG_ERROR << "Failed to load TensorRT Engine from " << trt_engine_file << "." << std::endl;
            return false;
        }

        FDUniquePtr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(*FDTrtLogger::Get())
        };
        if (!runtime) {
            MD_LOG_ERROR << "Failed to call createInferRuntime()." << std::endl;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_buffer.data(),
                                           engine_buffer.size()),
            FDInferDeleter());
        if (!engine_) {
            MD_LOG_ERROR << "Failed to call deserializeCudaEngine()." << std::endl;
            return false;
        }
        const int nb_bindings = engine_->getNbIOTensors();
        for (int i = 0; i < nb_bindings; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const nvinfer1::Dims dims = engine_->getTensorShape(name);
            const nvinfer1::DataType dtype = engine_->getTensorDataType(name);
            bool is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            std::string shape_str = vector_to_string(ToVec(dims));
            std::string type_str = datatype_to_string(GetFDDataType(dtype));
            if (is_input) {
                inputs_desc_.emplace_back(TrtValueInfo{name, ToVec(dims), dtype});
            }
            else {
                outputs_desc_.emplace_back(TrtValueInfo{name, ToVec(dims), dtype});
            }
        }
        context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        GetInputOutputInfo();

        for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
            auto const name = engine_->getIOTensorName(i);

            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }
            auto min = ToVec(engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX));
            auto max = ToVec(engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMIN));
            auto iter = shape_range_info_.find(name);
            if (iter == shape_range_info_.end()) {
                MD_LOG_ERROR << "There's no input named '" << name << "' in loaded model."
                    << std::endl;
                return false;
            }
            iter->second.Update(min);
            iter->second.Update(max);
        }
        MD_LOG_INFO << "Build TensorRT Engine from cache file: " << trt_engine_file
            << " with shape range information as below," << std::endl;
        for (const auto& item : shape_range_info_) {
            MD_LOG_INFO << item.second << std::endl;
        }
        return true;
    }

    bool TrtBackend::init(const RuntimeOption& option) {
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

        shape_range_info_.insert(std::make_pair("images", ShapeRangeInfo({1, 3, 640, 640})));
        const uint32_t flags =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) |
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

        builder_ = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(*FDTrtLogger::Get()));
        if (!builder_) {
            MD_LOG_ERROR << "Failed to call createInferBuilder()." << std::endl;
            return false;
        }
        network_ = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder_->createNetworkV2(flags));
        if (!network_) {
            MD_LOG_ERROR << "Failed to call createNetworkV2()." << std::endl;
            return false;
        }
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(*FDTrtLogger::Get()));
        std::cout << termcolor::green << "[TensorRT model buffer loaded, size: "
            << std::fixed << std::setprecision(3)
            << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]"
            << termcolor::reset << std::endl;
        // return init_from_onnx(model_buffer_, option_);
        return LoadTrtCache(trt_option.model_file);
    }


    bool TrtBackend::init_from_onnx(const std::string& model_buffer, const TrtBackendOption& option) {
        if (initialized_) {
            std::cerr << "TrtBackend is already initialized, cannot initialize again." << std::endl;
            return false;
        }
        try {
            // 1. 创建 ONNX Parser
            auto parser = nvonnxparser::createParser(*network_, *FDTrtLogger::Get());
            if (!parser->parse(model_buffer.c_str(), model_buffer.size())) {
                std::cerr << "Failed to parse ONNX model with TensorRT" << std::endl;
                return false;
            }
            // 2. 构建 engine
            auto config = builder_->createBuilderConfig();
            auto profile = builder_->createOptimizationProfile();
            for (const auto& item : shape_range_info_) {
                if (!profile->setDimensions(item.first.c_str(),
                                            nvinfer1::OptProfileSelector::kMIN,
                                            ToDims(item.second.min))) {
                    MD_LOG_FATAL << "[TrtBackend] Failed to set min_shape for input: " << item.first <<
                        " in TrtBackend." << std::endl;
                }

                if (!profile->setDimensions(item.first.c_str(),
                                            nvinfer1::OptProfileSelector::kMAX, ToDims(item.second.max))) {
                    MD_LOG_FATAL << "[TrtBackend] Failed to set max_shape for input: " << item.first <<
                        " in TrtBackend." << std::endl;
                }
                if (item.second.opt.size() == 0) {
                    if (!profile->setDimensions(item.first.c_str(),
                                                nvinfer1::OptProfileSelector::kOPT, ToDims(item.second.max))) {
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
                                               ToDims(item.second.opt))) {
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
        catch (const std::exception& e) {
            std::cerr << "Exception caught during TensorRT init: " << e.what() << std::endl;
            return false;
        }
    }

    int TrtBackend::ShapeRangeInfoUpdated(const std::vector<Tensor>& inputs) {
        bool need_update_engine = false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto iter = shape_range_info_.find(inputs[i].get_name());
            if (iter == shape_range_info_.end()) {
                MD_LOG_ERROR << "There's no input named '" << inputs[i].get_name() << "' in loaded model." << std::endl;
            }
            if (iter->second.Update(inputs[i].shape()) == 1) {
                need_update_engine = true;
            }
        }
        return need_update_engine;
    }

    bool TrtBackend::infer(std::vector<Tensor>& inputs,
                           std::vector<Tensor>* outputs) {
        if (inputs.size() != num_inputs()) {
            MD_LOG_ERROR << "Require " << num_inputs() << "inputs, but get " << inputs.size() << "." << std::endl;
            return false;
        }
        if (ShapeRangeInfoUpdated(inputs)) {
            // meet new shape output of predefined max/min shape
            // rebuild the tensorrt engine
            MD_LOG_WARN
                << "TensorRT engine will be rebuilt once shape range information "
                "changed, this may take lots of time, you can set a proper shape "
                "range before loading model to avoid rebuilding process. refer "
                "https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/"
                "faq/"
                "tensorrt_tricks.md for more details."
                << std::endl;
            BuildTrtEngine();
        }

        cudaSetDevice(option_.gpu_id);
        SetInputs(inputs);
        AllocateOutputsBuffer(outputs);

        if (!context_->executeV2(bindings_.data())) {
            MD_LOG_ERROR << "Failed to Infer with TensorRT." << std::endl;
            return false;
        }
        for (size_t i = 0; i < outputs->size(); ++i) {
            // if the final output tensor's dtype is different from the model output
            // tensor's dtype, then we need cast the data to the final output's dtype
            auto model_output_dtype =
                GetFDDataType(outputs_device_buffer_[(*outputs)[i].get_name()].dtype());
            casted_output_tensors_[(*outputs)[i].get_name()].from_external_memory(
                outputs_device_buffer_[(*outputs)[i].get_name()].data(),
                (*outputs)[i].shape(), model_output_dtype);
        }
        return true;
    }

    void TrtBackend::get_input_output_info() {
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
            bool is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            std::string shape_str = vector_to_string(ToVec(dims));
            std::string type_str = datatype_to_string(GetFDDataType(dtype));
            io_table.add_row({
                is_input ? "Input" : "Output",
                std::to_string(i),
                name,
                type_str,
                shape_str
            });
            if (is_input) {
                inputs_desc_.emplace_back(TrtValueInfo{name, ToVec(dims), dtype});
            }
            else {
                outputs_desc_.emplace_back(TrtValueInfo{name, ToVec(dims), dtype});
            }
        }
        std::cout << io_table << std::endl;
    }

    void TrtBackend::SetInputs(const std::vector<Tensor>& inputs) {
        for (const auto& item : inputs) {
            // auto idx = engine_->getBindingIndex(item.name.c_str());
            auto iter = io_name_index_.find(item.get_name());
            if (iter == io_name_index_.end()) {
                MD_LOG_FATAL << "TRTBackend SetInputs not find name: " << item.get_name() << std::endl;
            }
            auto idx = iter->second;
            auto const name = iter->first;
            std::vector<int> shape(item.shape().begin(), item.shape().end());
            auto dims = ToDims(shape);
            context_->setInputShape(name.c_str(), dims);

            // Allocate input buffer memory
            inputs_device_buffer_[item.get_name()].resize(dims);

            // copy from cpu to gpu
            if (item.dtype() == DataType::INT64) {
                int64_t* data = static_cast<int64_t*>(const_cast<void*>(item.data()));
                std::vector<int32_t> casted_data(data, data + item.size());
                // FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
                //                          static_cast<void*>(casted_data.data()),
                //                          item.Nbytes() / 2, cudaMemcpyHostToDevice,
                //                          stream_) == 0,
                //          "Error occurs while copy memory from CPU to GPU.");
                // WARN: For cudaMemcpyHostToDevice direction, cudaMemcpyAsync need page-locked host
                // memory to avoid any overlap to occur. The page-locked feature need by cudaMemcpyAsync
                // may not guarantee by FDTensor now. Reference:
                // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction
                if (!cudaMemcpy(inputs_device_buffer_[item.get_name()].data(),
                                static_cast<void*>(casted_data.data()),
                                item.byte_size() / 2, cudaMemcpyHostToDevice) == 0) {
                    MD_LOG_FATAL << "Error occurs while copy memory from CPU to GPU." << std::endl;
                }
            }
            else {
                // FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
                //                          item.Data(), item.Nbytes(),
                //                          cudaMemcpyHostToDevice, stream_) == 0,
                //          "Error occurs while copy memory from CPU to GPU.");
                // WARN: For cudaMemcpyHostToDevice direction, cudaMemcpyAsync need page-locked host
                // memory to avoid any overlap to occur. The page-locked feature need by cudaMemcpyAsync
                // may not guarantee by FDTensor now. Reference:
                // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction
                if (!cudaMemcpy(inputs_device_buffer_[item.get_name()].data(),
                                item.data(), item.byte_size(),
                                cudaMemcpyHostToDevice) == 0) {
                    MD_LOG_FATAL << "Error occurs while copy memory from CPU to GPU." << std::endl;
                }
            }

            // binding input buffer
            bindings_[idx] = inputs_device_buffer_[item.get_name()].data();
        }
    }

    void TrtBackend::AllocateOutputsBuffer(std::vector<Tensor>* outputs) {
        if (outputs->size() != outputs_desc_.size()) {
            outputs->resize(outputs_desc_.size());
        }
        for (size_t i = 0; i < outputs_desc_.size(); ++i) {
            // auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
            auto idx_iter = io_name_index_.find(outputs_desc_[i].name);
            if (idx_iter == io_name_index_.end()) {
                MD_LOG_FATAL << "TRTBackend Outputs not find name: " << outputs_desc_[i].name << std::endl;
            }

            auto idx = idx_iter->second;
            auto name = idx_iter->first;
            auto output_dims = context_->getTensorShape(name.c_str());

            // find the original index of output
            auto iter = outputs_order_.find(outputs_desc_[i].name);
            if (iter == outputs_order_.end()) {
                MD_LOG_FATAL << "Cannot find output: " << outputs_desc_[i].name <<
                    " of tensorrt network from the original model." << std::endl;
            }

            auto ori_idx = iter->second;

            // Allocate output buffer memory
            outputs_device_buffer_[outputs_desc_[i].name].resize(output_dims);

            // binding output buffer
            bindings_[idx] = outputs_device_buffer_[outputs_desc_[i].name].data();

            // set user's outputs info
            std::vector<int64_t> shape(output_dims.d,
                                       output_dims.d + output_dims.nbDims);
            // (*outputs)[ori_idx].is_pinned_memory = option_.enable_pinned_memory;
            (*outputs)[ori_idx].resize(shape, outputs_desc_[i].original_dtype,
                                       outputs_desc_[i].name);
        }
    }


    TensorInfo TrtBackend::get_input_info(int index) {
        if (index >= num_inputs()) {
            MD_LOG_FATAL << "The index: " << index << " should less than the number of inputs: " << num_inputs() << "."
                << std::endl;
        }

        TensorInfo info;
        info.name = inputs_desc_[index].name;
        info.shape.assign(inputs_desc_[index].shape.begin(),
                          inputs_desc_[index].shape.end());
        info.dtype = inputs_desc_[index].original_dtype;
        return info;
    }

    std::vector<TensorInfo> TrtBackend::get_input_infos() {
        std::vector<TensorInfo> infos;
        for (auto i = 0; i < inputs_desc_.size(); i++) {
            infos.emplace_back(get_input_info(i));
        }
        return infos;
    }

    TensorInfo TrtBackend::get_output_info(int index) {
        if (index >= num_outputs()) {
            MD_LOG_FATAL << "The index: " << index << " should less than the number of outputs: "
                << num_outputs() << "." << std::endl;
        }
        TensorInfo info;
        info.name = outputs_desc_[index].name;
        info.shape.assign(outputs_desc_[index].shape.begin(),
                          outputs_desc_[index].shape.end());
        info.dtype = outputs_desc_[index].original_dtype;
        return info;
    }

    std::vector<TensorInfo> TrtBackend::get_output_infos() {
        std::vector<TensorInfo> infos;
        for (auto i = 0; i < outputs_desc_.size(); i++) {
            infos.emplace_back(get_output_info(i));
        }
        return infos;
    }

    std::unique_ptr<BaseBackend> TrtBackend::clone(RuntimeOption& runtime_option,
                                                   void* stream, int device_id) {
        std::unique_ptr<BaseBackend> new_backend = std::make_unique<TrtBackend>();
        auto casted_backend = dynamic_cast<TrtBackend*>(new_backend.get());
        if (device_id > 0 && device_id != option_.gpu_id) {
            auto clone_option = option_;
            clone_option.gpu_id = device_id;
            clone_option.external_stream_ = stream;


            std::string model_buffer = "";
            if (!
                read_binary_from_file(clone_option.model_file, &model_buffer)) {
                MD_LOG_FATAL << "Fail to read binary from model file while cloning TrtBackend" << std::endl;
            }
            if (!casted_backend->init_from_onnx(model_buffer, clone_option)) {
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
        casted_backend->outputs_order_.insert(outputs_order_.begin(),
                                              outputs_order_.end());
        casted_backend->shape_range_info_.insert(shape_range_info_.begin(),
                                                 shape_range_info_.end());
        casted_backend->engine_ = engine_;
        casted_backend->context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            casted_backend->engine_->createExecutionContext());
        casted_backend->get_input_output_info();
        MD_LOG_INFO << "TRTBackend clone finish." << std::endl;
        return new_backend;
    }
} // namespace fastdeploy

//
// Created by aichao on 2025/6/20.
//

#include <MNN/Tensor.hpp>
#include "csrc/runtime/backends/mnn/mnn_backend.h"
#include "csrc/runtime/backends/mnn/utils.h"

namespace modeldeploy {
    MnnBackend::~MnnBackend() = default;

    bool MnnBackend::init(const RuntimeOption& runtime_option) {
        if (initialized_) {
            MD_LOG_ERROR << "MnnBackend is already initialized, cannot initialize again."
                << std::endl;
            return false;
        }
        if (!read_binary_from_file(runtime_option.model_file, &model_buffer_)) {
            MD_LOG_ERROR << "Failed to read model file: " << runtime_option.model_file << std::endl;
            return false;
        }
        const auto interpreter =
            MNN::Interpreter::createFromBuffer(model_buffer_.c_str(), model_buffer_.size());
        if (!interpreter) {
            MD_LOG_ERROR << "load mnn model file error, ensure model file is correct." << std::endl;
            return false;
        }
        net_.reset(interpreter, MNN::Interpreter::destroy);
        net_->setSessionMode(MNN::Interpreter::Session_Backend_Auto);
        net_->setSessionHint(MNN::Interpreter::MAX_TUNING_NUMBER, 5);
        build_option(runtime_option);
        const auto mnn_inputs = net_->getSessionInputAll(session_);
        const auto mnn_outputs = net_->getSessionOutputAll(session_);
        // 模型输入输出信息
        tabulate::Table input_table;
        input_table.format().font_color(tabulate::Color::yellow)
                   .border_color(tabulate::Color::blue)
                   .corner_color(tabulate::Color::blue);

        // input_table.add_row(Row_t{model_info_table});
        input_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
        input_table[0].format().font_style({tabulate::FontStyle::bold});
        for (auto& input : mnn_inputs) {
            TensorInfo info;
            info.name = input.first;
            info.shape = input.second->shape();
            info.dtype = mnn_dtype_to_md_dtype(input.second->getType());
            inputs_desc_.emplace_back(info);
            input_table.add_row({
                "Input",
                std::to_string(0),
                info.name,
                datatype_to_string(info.dtype),
                vector_to_string(info.shape)
            });
        }
        for (auto& output : mnn_outputs) {
            TensorInfo info;
            info.name = output.first;
            info.shape = output.second->shape();
            info.dtype = mnn_dtype_to_md_dtype(output.second->getType());
            outputs_desc_.emplace_back(info);
            input_table.add_row({
                "Output",
                std::to_string(0),
                info.name,
                datatype_to_string(info.dtype),
                vector_to_string(info.shape)
            });
        }
        std::cout << termcolor::green << "[model file:"
            << std::filesystem::absolute(runtime_option.model_file).filename().string()
            << " model size: " << std::fixed << std::setprecision(3)
            << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]"
            << termcolor::reset << std::endl;
        std::cout << input_table << std::endl;
        initialized_ = true;
        return true;
    }

    TensorInfo MnnBackend::get_input_info(int index) {
        if (index > num_inputs()) {
            MD_LOG_FATAL <<
                "The index: " << index << " should less than the number of inputs: "
                << num_inputs() << "." << std::endl;
        }

        return inputs_desc_[index];
    }

    //
    std::vector<TensorInfo> MnnBackend::get_input_infos() { return inputs_desc_; }

    TensorInfo MnnBackend::get_output_info(int index) {
        if (index > num_outputs()) {
            MD_LOG_FATAL <<
                "The index: " << index << " should less than the number of outputs: "
                << num_outputs() << "." << std::endl;
        }
        return outputs_desc_[index];
    }

    std::vector<TensorInfo> MnnBackend::get_output_infos() { return outputs_desc_; }

    bool MnnBackend::infer(std::vector<Tensor>& inputs,
                           std::vector<Tensor>* outputs) {
        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR << "[MnnBackend] Size of the inputs(" << inputs.size()
                << ") should keep same with the inputs of this model("
                << inputs_desc_.size() << ")." << std::endl;
            return false;
        }
        for (auto& input : inputs) {
            const auto tensor = net_->getSessionInput(session_, input.get_name().c_str());
            net_->resizeTensor(tensor, convert_shape<int64_t, int>(input.shape()));
            const auto mnn_tensor = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
            if (input.dtype() == DataType::FP32) {
                memcpy(mnn_tensor->host<float>(), input.data(), input.byte_size());
            }
            else if (input.dtype() == DataType::FP64) {
                memcpy(mnn_tensor->host<double>(), input.data(), input.byte_size());
            }
            else if (input.dtype() == DataType::INT8) {
                memcpy(mnn_tensor->host<int8_t>(), input.data(), input.byte_size());
            }
            else if (input.dtype() == DataType::INT64) {
                memcpy(mnn_tensor->host<int64_t>(), input.data(), input.byte_size());
            }
            else if (input.dtype() == DataType::INT32) {
                memcpy(mnn_tensor->host<int32_t>(), input.data(), input.byte_size());
            }
            else if (input.dtype() == DataType::UINT8) {
                memcpy(mnn_tensor->host<uint8_t>(), input.data(), input.byte_size());
            }
            else {
                MD_LOG_FATAL << "Unexpected data type of " << input.dtype() << std::endl;
            }
            tensor->copyFromHostTensor(mnn_tensor);
            MNN::Tensor::destroy(mnn_tensor);
        }
        net_->resizeSession(session_);
        net_->runSession(session_);
        for (size_t i = 0; i < outputs_desc_.size(); ++i) {
            auto tensor =
                net_->getSessionOutput(session_, outputs_desc_[i].name.c_str());
            if (outputs_desc_[i].dtype !=
                mnn_dtype_to_md_dtype(tensor->getType())) {
                outputs_desc_[i].dtype = mnn_dtype_to_md_dtype(tensor->getType());
            }

            outputs->resize(outputs_desc_.size());
            (*outputs)[i].allocate(convert_shape<int, int64_t>(tensor->shape()),
                                   outputs_desc_[i].dtype, outputs_desc_[i].name);
            // nchw data format
            const auto mnn_tensor = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
            tensor->copyToHostTensor(mnn_tensor);
            memcpy((*outputs)[i].data(), mnn_tensor->host<float>(), (*outputs)[i].byte_size());
            MNN::Tensor::destroy(mnn_tensor);
        }
        return true;
    }

    void MnnBackend::build_option(const RuntimeOption& option) {
        option_ = option.mnn_option;
        if (!option_.cache_file_path.empty()) {
            net_->setCacheFile(option_.cache_file_path.c_str());
        }
        MNN::ScheduleConfig config;
        MNN::BackendConfig backend_config;

        if (option.device == Device::CPU) {
            config.type = static_cast<MNNForwardType>(MNNForwardType::MNN_FORWARD_CPU);
            if (option_.cpu_thread_num > 0) {
                config.numThread = option_.cpu_thread_num;
            }
        }
        else if (option.device == Device::GPU) {
#if defined(__x86_64__) || defined(_M_X64)
            if (option_.forward_type != mnn::MNNForwardType::MNN_FORWARD_CUDA) {
                MD_LOG_WARN << "MnnBackend only support MNN_FORWARD_CUDA Format type, "
                    "switch to MNN_FORWARD_CUDA"
                    << std::endl;
            }
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_CUDA);
#elif defined(__aarch64__)
            config.type = static_cast<MNNForwardType>(option_.forward_type);
#else
            FDASSERT(false, "MnnBackend GPU only support aarch64 and x64 platform");
#endif
            if (option.device_id >= 0) {
                SET_MNN_GPU_ID(option.device_id);
            }
            config.mode = option_.gpu_mode;
            backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(option_.precision);
        }
        if (option_.power_mode != mnn::PowerMode::MNN_Power_Normal) {
#if defined(__aarch64__)
            backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(option_.power_mode);
#else
            MD_LOG_ERROR << "power mode of MNN_Power_High and MNN_Power_Low only be "
                "supported for aarch64 cpu, switch to MNN_Power_Normal"
                << std::endl;
#endif
        }
        backend_config.memory = static_cast<MNN::BackendConfig::MemoryMode>(option_.memory_mode);
        config.backendConfig = &backend_config;
        session_ = net_->createSession(config);
    }
} // namespace modeldeploy

//
// Created by aichao on 2025/6/26.
//

#define MNN_USER_SET_DEVICE
#include <MNN/MNNSharedContext.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "csrc/runtime/backends/mnn/mnn_module_api.h"
#include "csrc/runtime/backends/mnn/utils.h"


namespace modeldeploy {
    MnnBackend::~MnnBackend() = default;

    void MnnBackend::build_option(const RuntimeOption& option) {
        option_ = option.mnn_option;
        MNN::ScheduleConfig config;
        MNN::BackendConfig backend_config;
        MNNDeviceContext device_context;
        if (option.device == Device::CPU) {
            config.type = MNNForwardType::MNN_FORWARD_CPU;
            if (option_.cpu_thread_num > 0) {
                config.numThread = option_.cpu_thread_num;
            }
        }
        else if (option.device == Device::GPU) {
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_CUDA);
        }
        else if (option.device == Device::OPENCL) {
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_OPENCL);
        }
        else {
            MD_LOG_WARN << "Unsupported device: " << option.device << " switch to Auto." << std::endl;
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_AUTO);
        }
        if (option.device_id >= 0) {
            device_context.deviceId = option.device_id;
            backend_config.sharedContext = &device_context;
            // union 类型，如果是Device为CPU那么这里设置就会出错
            config.mode = option_.gpu_mode;
        }
        backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(option_.precision);
        if (option_.power_mode != mnn::PowerMode::MNN_Power_Normal) {
#if defined(__aarch64__)
            backend_config.power = static_cast<MNN::BackendConfig::PowerMode>(option_.power_mode);
#else
            MD_LOG_WARN << "power mode of MNN_Power_High and MNN_Power_Low only be "
                "supported for aarch64 cpu, switch to MNN_Power_Normal" << std::endl;
            option_.power_mode = mnn::PowerMode::MNN_Power_Normal;
#endif
        }
        backend_config.memory = static_cast<MNN::BackendConfig::MemoryMode>(option_.memory_mode);
        config.backendConfig = &backend_config;
        rtmgr_ = std::shared_ptr<
            MNN::Express::Executor::RuntimeManager>(
            MNN::Express::Executor::RuntimeManager::createRuntimeManager(config),
            MNN::Express::Executor::RuntimeManager::destroy);

        if (!option_.cache_file_path.empty()) {
            rtmgr_->setCache(option_.cache_file_path);
        }
        rtmgr_->setHint(MNN::Interpreter::GEOMETRY_COMPUTE_MASK, 0);
    }


    bool MnnBackend::init(const RuntimeOption& runtime_option) {
        if (initialized_) {
            MD_LOG_ERROR << "MnnBackend is already initialized, cannot initialize again."
                << std::endl;
            return false;
        }
        build_option(runtime_option);

        rtmgr_->setMode(MNN::Interpreter::Session_Release);


        if (!read_binary_from_file(runtime_option.model_file, &model_buffer_)) {
            MD_LOG_ERROR << "Failed to read model file: " << runtime_option.model_file << std::endl;
            return false;
        }

        net_ = std::shared_ptr<MNN::Express::Module>(
            MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{},
                                       reinterpret_cast<const uint8_t*>(model_buffer_.c_str()),
                                       model_buffer_.size(), rtmgr_), MNN::Express::Module::destroy);

        if (!net_) {
            MD_LOG_ERROR << "load mnn model file error, ensure model file is correct." << std::endl;
            return false;
        }

        const auto mnn_inputs = net_->getInfo()->inputs;
        const auto mnn_inputs_names = net_->getInfo()->inputNames;
        const auto mnn_outputs_names = net_->getInfo()->outputNames;

        // 模型输入输出信息
        tabulate::Table input_table;
        input_table.format().font_color(tabulate::Color::yellow)
                   .border_color(tabulate::Color::blue)
                   .corner_color(tabulate::Color::blue);

        // input_table.add_row(Row_t{model_info_table});
        input_table.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
        input_table[0].format().font_style({tabulate::FontStyle::bold});
        if (mnn_inputs.size() != mnn_inputs_names.size()) {
            MD_LOG_ERROR << "inputs size not equal to inputs names size." << std::endl;
            return false;
        }
        for (size_t i = 0; i < mnn_inputs.size(); ++i) {
            TensorInfo info;
            info.name = mnn_inputs_names[i];
            info.shape = mnn_inputs[i].dim;
            info.dtype = mnn_dtype_to_md_dtype(mnn_inputs[i].type);
            inputs_desc_.emplace_back(info);
            input_table.add_row({
                "Input",
                std::to_string(0),
                info.name,
                datatype_to_string(info.dtype),
                vector_to_string(info.shape)
            });
        }
        for (auto& output_name : mnn_outputs_names) {
            TensorInfo info;
            info.name = output_name;
            info.shape = {-1};
            info.dtype = DataType::UNKNOWN;
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
        std::vector<MNN::Express::VARP> mnn_inputs;
        for (auto& input : inputs) {
            auto tensor = MNN::Express::_Input(convert_shape<int64_t, int>(input.shape()),
                                               MNN::Express::NCHW,
                                               md_dtype_to_mnn_dtype(input.dtype()));
            tensor->setName(input.get_name());
            memcpy(tensor->writeMap<void>(), input.data(), input.byte_size());
            mnn_inputs.push_back(std::move(tensor));
        }
        const auto mnn_outputs = net_->onForward(mnn_inputs);
        if (mnn_outputs.size() != outputs_desc_.size()) {
            MD_LOG_ERROR << "[MnnBackend] Size of the outputs(" << mnn_outputs.size()
                << ") should keep same with the outputs of this model("
                << outputs_desc_.size() << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < outputs_desc_.size(); ++i) {
            outputs_desc_[i].dtype = mnn_dtype_to_md_dtype(mnn_outputs[i]->getInfo()->type);
            outputs_desc_[i].shape = mnn_outputs[i]->getInfo()->dim;
            outputs_desc_[i].name = mnn_outputs[i]->name();
            outputs->resize(outputs_desc_.size());
            (*outputs)[i].allocate(convert_shape<int, int64_t>(outputs_desc_[i].shape),
                                   outputs_desc_[i].dtype, outputs_desc_[i].name);
            memcpy((*outputs)[i].data(), mnn_outputs[i]->readMap<void>(), (*outputs)[i].byte_size());
        }
        return true;
    }


    std::map<std::string, std::string> MnnBackend::get_custom_meta_data() const {
        return net_->getInfo()->metaData;
    }
} // namespace modeldeploy

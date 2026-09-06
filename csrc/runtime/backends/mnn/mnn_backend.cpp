//
// Created by aichao on 2025/6/26.
//

#define MNN_USER_SET_DEVICE
#include <MNN/MNNSharedContext.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <tabulate/tabulate.hpp>
#include "core/md_log.h"
#include "runtime/backends/mnn/utils.h"
#include "runtime/backends/mnn/mnn_backend.h"
#include "runtime/backends/io_table.h"


namespace modeldeploy {
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
            config.mode = option_.gpu_mode;
        }
        else if (option.device == Device::OPENCL) {
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_OPENCL);
            config.mode = option_.gpu_mode;
        }
        else if (option.device == Device::VULKAN) {
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_VULKAN);
            config.mode = option_.gpu_mode;
        }
        else {
            MD_LOG_WARN << "Unsupported device: " << option.device << " switch to Auto." << std::endl;
            config.type = static_cast<MNNForwardType>(mnn::MNNForwardType::MNN_FORWARD_AUTO);
        }
        // 注意：MNN 的 sharedContext 语义按后端而异。OpenCL 把 sharedContext 当 MNNDeviceContext*
        // （读 deviceId/platformId 等），而 Vulkan 把它当 MNNVulkanContext*（读 pInstance/pDevice 等）。
        // 我们这里构造的是通用 MNNDeviceContext，若传给 Vulkan 会被强转成 MNNVulkanContext* 读到垃圾
        // -> Vulkan runtime 创建返回 nullptr -> 后续空指针崩溃。且 MNN Vulkan 本就把 GPU 硬编码为
        // tmpGpus[0]、device_id 无效。故 Vulkan（及 CPU）不设置 sharedContext。
        if (option.device_id >= 0 && option.device != Device::VULKAN) {
            device_context.deviceId = option.device_id;
            backend_config.sharedContext = &device_context;
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
        rtmgr_->setHint(MNN::Interpreter::GEOMETRY_COMPUTE_MASK, 0xFFFF);
    }


    bool MnnBackend::init(const RuntimeOption& runtime_option) {
        const_cast<RuntimeOption&>(runtime_option).validate();
        if (initialized_) {
            MD_LOG_ERROR << "MnnBackend is already initialized, cannot initialize again."
                << std::endl;
            return false;
        }
        build_option(runtime_option);

        rtmgr_->setMode(MNN::Interpreter::Session_Release);
        if (runtime_option.model_from_memory) {
            model_buffer_ = runtime_option.model_buffer;
        }
        else {
            if (!read_binary_from_file(runtime_option.model_file, &model_buffer_)) {
                MD_LOG_ERROR << "Failed to read model file: " << runtime_option.model_file << std::endl;
                return false;
            }
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
        }
        for (auto& output_name : mnn_outputs_names) {
            TensorInfo info;
            info.name = output_name;
            info.shape = {-1};
            info.dtype = DataType::UNKNOWN;
            outputs_desc_.emplace_back(info);
        }
        // init 时 MNN 输出形状未知，跑一次零数据 dummy forward 补全（动态维用 1 占位）。
        infer_output_info();
        MD_LOG_INFO
            << "[model file:"
            << std::filesystem::absolute(runtime_option.model_file).filename().string()
            << " model size: " << std::fixed << std::setprecision(3)
            << static_cast<float>(model_buffer_.size()) / 1024 / 1024.0f << "MB]"
            << std::endl;
        MD_LOG_INFO << std::endl << build_io_table(inputs_desc_, outputs_desc_) << std::endl;
        initialized_ = true;
        return true;
    }

    void MnnBackend::infer_output_info() {
        try {
            const auto& mnn_inputs = net_->getInfo()->inputs;
            if (mnn_inputs.size() != inputs_desc_.size()) return;
            const auto& mnn_outputs_names = net_->getInfo()->outputNames;
            std::vector<MNN::Express::VARP> probe_inputs;
            probe_inputs.reserve(mnn_inputs.size());
            for (size_t i = 0; i < mnn_inputs.size(); ++i) {
                std::vector<int> dims = mnn_inputs[i].dim;
                for (auto& d : dims) if (d <= 0) d = 1; // 动态维占位
                auto v = MNN::Express::_Input(dims, MNN::Express::NCHW, mnn_inputs[i].type);
                v->setName(inputs_desc_[i].name);
                probe_inputs.push_back(v);
            }
            const auto outs = net_->onForward(probe_inputs);
            if (outs.size() != outputs_desc_.size()) return;
            for (size_t i = 0; i < outs.size(); ++i) {
                const auto* info = outs[i]->getInfo();
                if (!info) continue;
                outputs_desc_[i].shape = convert_shape<int, int>(info->dim);
                outputs_desc_[i].dtype = mnn_dtype_to_md_dtype(info->type);
                outputs_desc_[i].name = mnn_outputs_names[i];
            }
        }
        catch (...) {
            MD_LOG_WARN << "[MnnBackend] init output-shape probe failed; keep [-1]." << std::endl;
        }
    }

    TensorInfo MnnBackend::get_input_info(int index) {
        if (index < 0 || index >= num_inputs()) {
            MD_LOG_FATAL <<
                "The index: " << index << " should less than the number of inputs: "
                << num_inputs() << "." << std::endl;
        }
        return inputs_desc_[index];
    }

    std::vector<TensorInfo> MnnBackend::get_input_infos() { return inputs_desc_; }

    TensorInfo MnnBackend::get_output_info(const int index) {
        if (index < 0 || index >= num_outputs()) {
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
        std::vector<MNN::Express::VARP> mnn_inputs(inputs.size());
        if (cached_inputs_.size() != inputs.size()) {
            cached_inputs_.resize(inputs.size());
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            auto& tens = cached_inputs_[i];
            bool shape_changed = (tens == nullptr) ||
                tens->getInfo()->dim != convert_shape<int64_t, int>(input.shape());
            bool dtype_changed = (tens == nullptr) ||
                tens->getInfo()->type != md_dtype_to_mnn_dtype(input.dtype());
            if (shape_changed || dtype_changed) {
                tens = MNN::Express::_Input(convert_shape<int64_t, int>(input.shape()),
                                            MNN::Express::NCHW,
                                            md_dtype_to_mnn_dtype(input.dtype()));
                tens->setName(input.get_name());
            }
            // 复用同一输入 VARP（MNN 图缓存按输入身份缓存，换新输入会触发 GPU 重排，
            // 实测 OpenCL 慢约一倍）。仅数据内容更新，shape/dtype 不变则复用缓存。
            memcpy(tens->writeMap<void>(), input.data(), input.byte_size());
            mnn_inputs[i] = tens;
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
                                   outputs_desc_[i].dtype, Device::CPU, outputs_desc_[i].name);
            memcpy((*outputs)[i].data(), mnn_outputs[i]->readMap<void>(), (*outputs)[i].byte_size());
        }
        return true;
    }


    std::map<std::string, std::string> MnnBackend::get_custom_meta_data() const {
        return net_->getInfo()->metaData;
    }

    std::unique_ptr<BaseBackend> MnnBackend::clone(const RuntimeOption& runtime_option,
                                                   void* stream, int device_id) {
        // 真共享克隆：不重载模型、不复制权重。直接共享已加载的 net_（MNN::Express::Module，
        // 持有网络图与权重所在的 RuntimeManager 后端内存）与 rtmgr_，即共享同一份模型/设备内存。
        // 仅输入输出描述为各实例独立。
        (void)runtime_option;
        (void)stream;
        (void)device_id;
        if (!net_) return nullptr;
        auto nb = std::make_unique<MnnBackend>();
        nb->rtmgr_ = rtmgr_;
        nb->net_ = net_;
        nb->model_buffer_ = model_buffer_;
        nb->option_ = option_;
        nb->inputs_desc_ = inputs_desc_;
        nb->outputs_desc_ = outputs_desc_;
        nb->initialized_ = true;
        return nb;
    }
} // namespace modeldeploy

//
// Created by aichao on 2025/8/2.
// Sophgo 算能 TPU 推理后端：sail::Engine（SYSIO 模式）加载 .bmodel。
// 适配 sail 3.11（BM1688 SOC）：Engine(bmodel, tpu_id, IOMode) 构造、
// create_input/output_tensors_map + process(graph_name) 推理。
//

#include "core/md_log.h"
#include "runtime/backends/sophgo/sophgo_backend.h"

#ifdef ENABLE_SOPHGO
#include "sail/engine.h"
#include "sail/tensor.h"
#include <cstring>

namespace modeldeploy {
namespace {
    // bmodel 的 shape 是 vector<int>，直接可用
    DataType sail_dtype_to_md(const bm_data_type_t t) {
        switch (t) {
            case BM_FLOAT32: return DataType::FP32;
            case BM_FLOAT16: return DataType::FP32; // 统一转 FP32 推理结果
            case BM_BFLOAT16: return DataType::FP32;
            case BM_INT32: return DataType::INT32;
            case BM_UINT32: return DataType::INT32;
            case BM_INT16: return DataType::INT32;
            case BM_UINT16: return DataType::INT32;
            case BM_UINT8: return DataType::UINT8;
            case BM_INT8: return DataType::INT8;
            default: return DataType::UNKNOWN;
        }
    }
} // namespace

    SophgoBackend::~SophgoBackend() {
        if (engine_) delete static_cast<sail::Engine*>(engine_);
        engine_ = nullptr;
        handle_ = nullptr;
    }

    bool SophgoBackend::init(const RuntimeOption& option) {
        if (initialized_) {
            MD_LOG_ERROR << "SophgoBackend is already initialized." << std::endl;
            return false;
        }
        bmodel_path_ = option.sophgo_option.bmodel_path.empty()
            ? option.model_file : option.sophgo_option.bmodel_path;
        const int device_id = option.sophgo_option.device_id;

        // sail 3.11：Engine(bmodel_path, tpu_id, IOMode)；SYSIO = 输入/输出均在系统内存
        auto* engine = new sail::Engine(bmodel_path_, device_id, sail::IOMode::SYSIO);
        const auto graphs = engine->get_graph_names();
        if (graphs.empty()) {
            MD_LOG_ERROR << "No graph found in bmodel: " << bmodel_path_ << std::endl;
            delete engine;
            return false;
        }
        graph_name_ = graphs[0];

        // 输入输出描述
        const auto input_names = engine->get_input_names(graph_name_);
        const auto output_names = engine->get_output_names(graph_name_);
        for (const auto& n : input_names) {
            TensorInfo info;
            info.name = n;
            info.shape = engine->get_input_shape(graph_name_, n);
            info.dtype = sail_dtype_to_md(engine->get_input_dtype(graph_name_, n));
            inputs_desc_.emplace_back(std::move(info));
        }
        for (const auto& n : output_names) {
            TensorInfo info;
            info.name = n;
            info.shape = engine->get_output_shape(graph_name_, n);
            info.dtype = sail_dtype_to_md(engine->get_output_dtype(graph_name_, n));
            outputs_desc_.emplace_back(std::move(info));
        }

        engine_ = engine;
        initialized_ = true;
        MD_LOG_INFO << "SophgoBackend loaded " << bmodel_path_
            << " graph[" << graph_name_ << "] inputs=" << inputs_desc_.size()
            << " outputs=" << outputs_desc_.size() << std::endl;
        return true;
    }

    bool SophgoBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (!initialized_ || !engine_) return false;
        auto* engine = static_cast<sail::Engine*>(engine_);

        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR << "[SophgoBackend] inputs size mismatch: " << inputs.size()
                << " vs " << inputs_desc_.size() << std::endl;
            return false;
        }

        // sail 3.11：创建内置输入/输出 tensor map（SYSIO 模式，sys_data 可直接 memcpy）
        auto input_map = engine->create_input_tensors_map(graph_name_);
        auto output_map = engine->create_output_tensors_map(graph_name_);

        // 写输入（host Tensor -> sail::Tensor sys_data -> sync_s2d）
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto it = input_map.find(inputs_desc_[i].name);
            if (it == input_map.end() || it->second == nullptr) {
                MD_LOG_ERROR << "[SophgoBackend] input not found: " << inputs_desc_[i].name << std::endl;
                return false;
            }
            sail::Tensor* st = it->second;
            const size_t bytes = inputs[i].byte_size();
            if (bytes > static_cast<size_t>(st->size() * st->element_size())) {
                MD_LOG_ERROR << "[SophgoBackend] input size too large: " << inputs_desc_[i].name << std::endl;
                return false;
            }
            std::memcpy(st->sys_data(), inputs[i].data(), bytes);
            st->sync_s2d();
        }

        engine->process(graph_name_, input_map, output_map);

        // 读输出（sail::Tensor sys_data <- sync_d2s -> host Tensor）
        outputs->resize(outputs_desc_.size());
        for (size_t i = 0; i < outputs_desc_.size(); ++i) {
            auto it = output_map.find(outputs_desc_[i].name);
            if (it == output_map.end() || it->second == nullptr) {
                MD_LOG_ERROR << "[SophgoBackend] output not found: " << outputs_desc_[i].name << std::endl;
                return false;
            }
            sail::Tensor* st = it->second;
            st->sync_d2s();
            const auto shape = st->shape();  // vector<int>
            const std::vector<int64_t> shape64(shape.begin(), shape.end());
            outputs_desc_[i].shape = shape;
            outputs_desc_[i].dtype = sail_dtype_to_md(st->dtype());
            (*outputs)[i].allocate(shape64, outputs_desc_[i].dtype, Device::CPU, outputs_desc_[i].name);
            std::memcpy((*outputs)[i].data(), st->sys_data(), (*outputs)[i].byte_size());
        }
        return true;
    }

    std::unique_ptr<BaseBackend> SophgoBackend::clone(const RuntimeOption& runtime_option,
                                                       void* stream, int device_id) {
        (void)stream;
        auto nb = std::make_unique<SophgoBackend>();
        RuntimeOption opt = runtime_option;
        if (device_id >= 0) {
            opt.sophgo_option.device_id = device_id;
        }
        if (!nb->init(opt)) {
            return nullptr;
        }
        return nb;
    }

    TensorInfo SophgoBackend::get_input_info(const int index) {
        return inputs_desc_[index];
    }

    TensorInfo SophgoBackend::get_output_info(const int index) {
        return outputs_desc_[index];
    }

    std::vector<TensorInfo> SophgoBackend::get_input_infos() { return inputs_desc_; }
    std::vector<TensorInfo> SophgoBackend::get_output_infos() { return outputs_desc_; }

    std::map<std::string, std::string> SophgoBackend::get_custom_meta_data() const {
        return {};
    }
} // namespace modeldeploy

#else // !ENABLE_SOPHGO

namespace modeldeploy {
    SophgoBackend::~SophgoBackend() = default;
    bool SophgoBackend::init(const RuntimeOption& option) {
        (void)option;
        MD_LOG_FATAL << "SophgoBackend is not available, please compiled with ENABLE_SOPHGO=ON."
            << std::endl;
        return false;
    }
    bool SophgoBackend::infer(std::vector<Tensor>&, std::vector<Tensor>*) { return false; }
    std::unique_ptr<BaseBackend> SophgoBackend::clone(const RuntimeOption&, void*, int) {
        return nullptr;
    }
    TensorInfo SophgoBackend::get_input_info(const int index) { return inputs_desc_[index]; }
    TensorInfo SophgoBackend::get_output_info(const int index) { return outputs_desc_[index]; }
    std::vector<TensorInfo> SophgoBackend::get_input_infos() { return inputs_desc_; }
    std::vector<TensorInfo> SophgoBackend::get_output_infos() { return outputs_desc_; }
    std::map<std::string, std::string> SophgoBackend::get_custom_meta_data() const { return {}; }
} // namespace modeldeploy

#endif // ENABLE_SOPHGO

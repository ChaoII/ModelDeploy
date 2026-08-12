//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "runtime/backends/backend.h"
#include "runtime/backends/sophgo/option.h"

namespace modeldeploy {
    class SophgoBackend : public BaseBackend {
    public:
        SophgoBackend() = default;
        ~SophgoBackend() override;

        bool init(const RuntimeOption& option) override;

        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;

        std::unique_ptr<BaseBackend> clone(const RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

        [[nodiscard]] size_t num_inputs() const override { return inputs_desc_.size(); }
        [[nodiscard]] size_t num_outputs() const override { return outputs_desc_.size(); }

        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const override;

    private:
        std::string bmodel_path_;
        std::string graph_name_;
        // 不透明句柄：bmrt context (void*)、bm_handle_t (void*)、bm_net_info_t* (const void*)
        void* bmrt_ = nullptr;
        void* handle_ = nullptr;
        const void* net_info_ = nullptr;
        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
        // 缓存推理输入/输出设备内存（bm_device_mem_t 数组，首次分配、复用、析构释放），
        // 避免每次 bm_free_device_mem 导致 bmrt 状态异常/段错误
        void* cached_in_mems_ = nullptr;
        void* cached_out_mems_ = nullptr;
        bool io_cached_ = false;
        // bm_misc_info（SOC/PCIe 模式判断，mmap 读输出用），不透明句柄
        void* misc_info_ = nullptr;
        bool ensure_io_cache();
    };
} // namespace modeldeploy

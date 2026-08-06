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

        // 返回缓存的输入设备内存（bm_device_mem_t*，bmrt_tensor 分配并缓存复用），
        // 供 SophgoProcessorBackend 的 BMCV 预处理 attach 写入（零拷贝输入）。
        // 未初始化时为 nullptr；首次调用 lazily 分配。生命周期由 backend 管理。
        // 零拷贝输入经 Tensor::from_external_memory(..., Device::TPU) 包装后走统一 infer()，
        // infer() 识别 Device::TPU 输入跳过 s2d 上传直接 launch。
        void* get_input_device_mem();

        // 返回 bmrt 内部 bm_handle_t（供 SophgoProcessorBackend 共享，D2D 零拷贝需要同一 handle）
        [[nodiscard]] void* get_bm_handle();

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

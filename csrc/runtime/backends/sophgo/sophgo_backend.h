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
        // 已加载的 TPU bmodel 资源（bmrt context + 设备 handle + 网络信息）。
        // clone() 共享同一 `Engine`，析构时最后一个引用才 bmrt_destroy + bm_dev_free——
        // 即多个 clone 真实共享同一份 TPU 权重/算子内存，绝不重载、不复制。
        struct Engine {
            void* bmrt = nullptr;            // bmrt context
            void* handle = nullptr;          // bm_handle_t
            const void* net_info = nullptr;  // bm_net_info_t*（指向 bmrt 内部，随 bmrt 存活）
            std::string bmodel_path;
            std::string graph_name;
            Engine() = default;
            ~Engine();
        };
        std::shared_ptr<Engine> engine_;
        // 推理输入/输出设备内存（bm_device_mem_t 数组，首次分配、复用、析构释放）。
        // 每个实例各自独立缓存，避免 clone 并发 infer 相互覆盖；内存由共享 bmrt 统一释放。
        void* cached_in_mems_ = nullptr;
        void* cached_out_mems_ = nullptr;
        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
        bool io_cached_ = false;
        // bm_misc_info（SOC/PCIe 模式判断，mmap 读输出用），不透明句柄
        void* misc_info_ = nullptr;
        bool ensure_io_cache();
    };
} // namespace modeldeploy

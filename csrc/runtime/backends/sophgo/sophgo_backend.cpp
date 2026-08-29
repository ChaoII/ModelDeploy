//
// Created by aichao on 2025/8/2.
// Sophgo 算能 TPU 推理后端：libsophon bmrt 直接推理（替代 sail）。
// 零拷贝：BMCV 预处理结果写入 bmrt_tensor 分配的输入设备内存，调用方用
// Tensor::from_external_memory(..., Device::TPU) 包装后走统一 infer()，
// infer() 识别 TPU 输入跳过 s2d 直接 launch，配合 BMCV 设备预处理，跳过 CPU 往返拷贝。
//

#include "core/md_log.h"
#include "runtime/backends/sophgo/sophgo_backend.h"

#include "bmlib_runtime.h"
#include "bmdef.h"
#include "bmruntime_interface.h"
#include <cstring>

namespace modeldeploy {
namespace {
    DataType bm_dtype_to_md(const bm_data_type_t t) {
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

    std::vector<int> bm_shape_to_vec(const bm_shape_t& s) {
        std::vector<int> v(s.dims, s.dims + s.num_dims);
        return v;
    }
} // namespace

    SophgoBackend::Engine::~Engine() {
        // 缓存的 input/output 设备内存由 bmrt_tensor 分配，bmrt_destroy 统一释放。
        // 多个 clone 共享同一 engine_，仅在最后一个引用释放时才销毁 bmrt/handle（TPU 内存）。
        if (bmrt) {
            bmrt_destroy(static_cast<void*>(bmrt));
            bmrt = nullptr;
        }
        if (handle) {
            bm_dev_free(static_cast<bm_handle_t>(handle));
            handle = nullptr;
        }
        net_info = nullptr;
    }

    SophgoBackend::~SophgoBackend() {
        // 只释放本实例自己的 io 缓存数组描述（设备内存仍由共享 engine_->bmrt 统一释放）。
        if (cached_in_mems_) {
            delete[] static_cast<bm_device_mem_t*>(cached_in_mems_);
            cached_in_mems_ = nullptr;
        }
        if (cached_out_mems_) {
            delete[] static_cast<bm_device_mem_t*>(cached_out_mems_);
            cached_out_mems_ = nullptr;
        }
        io_cached_ = false;
        delete static_cast<bm_misc_info*>(misc_info_);
        misc_info_ = nullptr;
        // engine_ 为 shared_ptr：最后一个引用析构时自动 bmrt_destroy + bm_dev_free。
        engine_.reset();
    }

    bool SophgoBackend::init(const RuntimeOption& option) {
        if (initialized_) {
            MD_LOG_ERROR << "SophgoBackend is already initialized." << std::endl;
            return false;
        }
        const_cast<RuntimeOption&>(option).validate();
        auto engine = std::make_shared<SophgoBackend::Engine>();
        engine->bmodel_path = option.sophgo_option.bmodel_path.empty()
            ? option.model_file : option.sophgo_option.bmodel_path;
        // device_id 默认 -1（use_sophgo_backend 只设 TPU 不动它），bm_dev_request(-1) 会失败，
        // 此处兜底为 0 号设备，避免“默认配置无法初始化”的坑。
        const int device_id = (option.device_id >= 0) ? option.device_id
                              : (option.sophgo_option.device_id >= 0 ? option.sophgo_option.device_id : 0);

        bm_handle_t h = nullptr;
        if (bm_dev_request(&h, device_id) != BM_SUCCESS) {
            MD_LOG_ERROR << "[SophgoBackend] bm_dev_request failed (device " << device_id << ")." << std::endl;
            return false;
        }
        engine->handle = static_cast<void*>(h);
        // 每实例各自持有 bm_misc_info，避免进程级 static 被多实例/多线程争写读到陈旧值。
        delete static_cast<bm_misc_info*>(misc_info_);
        auto* m = new bm_misc_info{};
        bm_get_misc_info(h, m);
        misc_info_ = m;

        void* bmrt = bmrt_create(h);
        if (!bmrt) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_create failed." << std::endl;
            bm_dev_free(h);
            return false;
        }
        if (!bmrt_load_bmodel(bmrt, engine->bmodel_path.c_str())) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_load_bmodel failed: " << engine->bmodel_path << std::endl;
            bmrt_destroy(bmrt);
            bm_dev_free(h);
            return false;
        }
        engine->bmrt = bmrt;

        const char** net_names = nullptr;
        bmrt_get_network_names(bmrt, &net_names);
        if (!net_names || !net_names[0]) {
            MD_LOG_ERROR << "[SophgoBackend] no network in bmodel." << std::endl;
            return false;
        }
        engine->graph_name = net_names[0];
        const bm_net_info_t* info = bmrt_get_network_info(bmrt, engine->graph_name.c_str());
        if (!info) {
            MD_LOG_ERROR << "[SophgoBackend] get_network_info failed: " << engine->graph_name << std::endl;
            return false;
        }
        engine->net_info = info;

        for (int i = 0; i < info->input_num; ++i) {
            TensorInfo ti;
            ti.name = info->input_names[i];
            ti.shape = bm_shape_to_vec(info->stages[0].input_shapes[i]);
            ti.dtype = bm_dtype_to_md(info->input_dtypes[i]);
            inputs_desc_.emplace_back(std::move(ti));
        }
        for (int i = 0; i < info->output_num; ++i) {
            TensorInfo ti;
            ti.name = info->output_names[i];
            ti.shape = bm_shape_to_vec(info->stages[0].output_shapes[i]);
            ti.dtype = bm_dtype_to_md(info->output_dtypes[i]);
            outputs_desc_.emplace_back(std::move(ti));
        }

        engine_ = std::move(engine);
        initialized_ = true;
        MD_LOG_INFO << "SophgoBackend(bmrt) loaded " << engine_->bmodel_path
            << " graph[" << engine_->graph_name << "] inputs=" << inputs_desc_.size()
            << " outputs=" << outputs_desc_.size() << std::endl;
        return true;
    }

    bool SophgoBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (!initialized_ || !engine_ || !engine_->bmrt || !engine_->net_info) return false;
        bm_handle_t h = static_cast<bm_handle_t>(engine_->handle);
        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR << "[SophgoBackend] inputs size mismatch: " << inputs.size()
                << " vs " << inputs_desc_.size() << std::endl;
            return false;
        }
        const bm_net_info_t* info = static_cast<const bm_net_info_t*>(engine_->net_info);
        const size_t ni = inputs.size();
        const size_t no = outputs_desc_.size();

        std::vector<bm_tensor_t> in_t(ni), out_t(no);
        // 缓存 io 设备内存（首次 bmrt_tensor 分配并缓存，之后复用，避免每次 free 的稳定性问题）
        if (!io_cached_ || !cached_in_mems_ || !cached_out_mems_) {
            if (!ensure_io_cache()) return false;
        }
        bm_device_mem_t* ins = static_cast<bm_device_mem_t*>(cached_in_mems_);
        bm_device_mem_t* outs = static_cast<bm_device_mem_t*>(cached_out_mems_);
        for (size_t i = 0; i < no; ++i) {
            out_t[i].device_mem = outs[i];
            out_t[i].dtype = info->output_dtypes[i];
            out_t[i].shape = info->stages[0].output_shapes[i];
        }

        for (size_t i = 0; i < ni; ++i) {
            in_t[i].dtype = info->input_dtypes[i];
            in_t[i].shape = info->stages[0].input_shapes[i];
            if (inputs[i].device() == Device::TPU && !inputs[i].get_owns_data()) {
                // 零拷贝输入：Tensor 持有 BMCV 写入的设备内存（bm_device_mem_t*），
                // 直接作为输入内存 launch，跳过 s2d 上传。
                const bm_device_mem_t* dev =
                    static_cast<const bm_device_mem_t*>(inputs[i].data());
                if (!dev) {
                    MD_LOG_ERROR << "[SophgoBackend] TPU input tensor data is null." << std::endl;
                    return false;
                }
                in_t[i].device_mem = *dev;
            } else {
                // 常规路径：上传到缓存的输入设备内存
                in_t[i].device_mem = ins[i];
                if (bm_memcpy_s2d(h, in_t[i].device_mem, inputs[i].data()) != BM_SUCCESS) {
                    MD_LOG_ERROR << "[SophgoBackend] bm_memcpy_s2d(input) failed." << std::endl;
                    return false;
                }
            }
        }

        if (!bmrt_launch_tensor_ex(engine_->bmrt, engine_->graph_name.c_str(),
                                   in_t.data(), static_cast<int>(ni),
                                   out_t.data(), static_cast<int>(no),
                                   /*user_mem*/true, /*user_stmode*/false)) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_launch_tensor failed." << std::endl;
            return false;
        }
        bm_thread_sync(h);

        outputs->resize(no);
        // SOC 模式用 mmap 零拷贝读输出，PCIe 用 d2s 拷贝
        const bm_misc_info* mi = static_cast<const bm_misc_info*>(misc_info_);
        const bool is_soc = mi && mi->pcie_soc_mode == 1;
        for (size_t i = 0; i < no; ++i) {
            // 输出 shape/dtype 以 bmodel 静态信息为准（bmrt_launch_tensor 可能改写 out_t[i]）
            const auto os = info->stages[0].output_shapes[i];
            std::vector<int64_t> shape64(os.dims, os.dims + os.num_dims);
            const bm_data_type_t od = info->output_dtypes[i];
            outputs_desc_[i].dtype = bm_dtype_to_md(od);
            (*outputs)[i].allocate(shape64, DataType::FP32, Device::CPU, outputs_desc_[i].name);
            const uint8_t* src = nullptr;
            std::vector<uint8_t> scratch;
            if (is_soc) {
                unsigned long long addr = 0;
                if (bm_mem_mmap_device_mem(h, &out_t[i].device_mem, &addr) != BM_SUCCESS ||
                    bm_mem_invalidate_device_mem(h, &out_t[i].device_mem) != BM_SUCCESS) {
                    MD_LOG_ERROR << "[SophgoBackend] mmap output failed." << std::endl;
                    return false;
                }
                scratch.assign(reinterpret_cast<uint8_t*>(addr),
                               reinterpret_cast<uint8_t*>(addr) +
                                   static_cast<size_t>(bm_mem_get_device_size(out_t[i].device_mem)));
                bm_mem_unmap_device_mem(h, reinterpret_cast<void*>(addr),
                                        bm_mem_get_device_size(out_t[i].device_mem));
                src = scratch.data();
            } else {
                scratch.resize(static_cast<size_t>(bm_mem_get_device_size(out_t[i].device_mem)));
                if (bm_memcpy_d2s(h, scratch.data(), out_t[i].device_mem) != BM_SUCCESS) {
                    MD_LOG_ERROR << "[SophgoBackend] bm_memcpy_d2s(output) failed." << std::endl;
                    return false;
                }
                src = scratch.data();
            }
            // 统一成 FP32 结果（Data 层无 FP16/BF16 类型）。SOC/PIce 均先取原始字节再转主机。
            float* dst = reinterpret_cast<float*>((*outputs)[i].data());
            if (od == BM_FLOAT16) {
                const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src);
                const size_t n = bm_mem_get_device_size(out_t[i].device_mem) / 2;
                for (size_t k = 0; k < n; ++k) {
                    const uint32_t u = p16[k];
                    const uint32_t s = (u & 0x8000u) << 16;
                    const uint32_t e = (u & 0x7c00u);
                    const uint32_t m = (u & 0x03ffu);
                    uint32_t bits = 0;
                    if (e == 0) {  // 零或次正规
                        if (m == 0) bits = s;
                        else {  // 次正规 → 正规
                            uint32_t mm = m;
                            int re = 127 - 15 + 1;
                            while (!(mm & 0x400u)) { mm <<= 1; --re; }
                            bits = s | (static_cast<uint32_t>(re) << 23) | ((mm & 0x3ffu) << 13);
                        }
                    } else if (e == 0x7c00u) {  // inf / nan
                        bits = s | 0x7f800000u | (m << 13);
                    } else {  // 正规
                        bits = s | ((e + (127 - 15)) << 23) | (m << 13);
                    }
                    dst[k] = *reinterpret_cast<float*>(&bits);
                }
            } else if (od == BM_BFLOAT16) {
                const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src);
                const size_t n = bm_mem_get_device_size(out_t[i].device_mem) / 2;
                for (size_t k = 0; k < n; ++k) {
                    const uint32_t bits = static_cast<uint32_t>(p16[k]) << 16;
                    dst[k] = *reinterpret_cast<const float*>(&bits);
                }
            } else {
                // FP32/int8 等原始字节数与 FP32 输出字节数一致（int8 亦按 numel 扩展，见下）
                const size_t raw = static_cast<size_t>(bm_mem_get_device_size(out_t[i].device_mem));
                if (raw >= (*outputs)[i].byte_size()) {
                    memcpy(dst, src, (*outputs)[i].byte_size());
                } else {
                    memcpy(dst, src, raw);
                }
            }
        }
        return true;
    }

    bool SophgoBackend::ensure_io_cache() {
        if (io_cached_ && cached_in_mems_ && cached_out_mems_) return true;
        if (!engine_ || !engine_->bmrt || !engine_->net_info) return false;
        const bm_net_info_t* info = static_cast<const bm_net_info_t*>(engine_->net_info);
        const size_t ni = inputs_desc_.size();
        const size_t no = outputs_desc_.size();
        auto* ins = new bm_device_mem_t[ni];
        for (size_t i = 0; i < ni; ++i) {
            bm_tensor_t t;
            if (!bmrt_tensor(&t, engine_->bmrt, info->input_dtypes[i],
                             info->stages[0].input_shapes[i])) {
                MD_LOG_ERROR << "[SophgoBackend] bmrt_tensor(input) failed." << std::endl;
                delete[] ins;
                return false;
            }
            ins[i] = t.device_mem;
        }
        auto* outs = new bm_device_mem_t[no];
        for (size_t i = 0; i < no; ++i) {
            bm_tensor_t t;
            if (!bmrt_tensor(&t, engine_->bmrt, info->output_dtypes[i],
                             info->stages[0].output_shapes[i])) {
                MD_LOG_ERROR << "[SophgoBackend] bmrt_tensor(output) failed." << std::endl;
                delete[] ins; delete[] outs;
                return false;
            }
            outs[i] = t.device_mem;
        }
        cached_in_mems_ = ins;
        cached_out_mems_ = outs;
        io_cached_ = true;
        return true;
    }

    std::unique_ptr<BaseBackend> SophgoBackend::clone(const RuntimeOption& runtime_option,
                                                       void* stream, int device_id) {
        // 真共享克隆：不与原实例重建 TPU engine（不重载 bmodel、不复制权重）。
        // clone 直接共享已加载的 engine_（bmrt/handle/net_info），即共享同一份 TPU 权重/算子内存；
        // 仅输入输出描述与 io 设备内存缓存为各实例独立（避免并发 infer 相互覆盖）。
        (void)stream;
        (void)device_id;
        (void)runtime_option;
        if (!engine_) return nullptr;
        auto nb = std::make_unique<SophgoBackend>();
        nb->engine_ = engine_;
        nb->inputs_desc_ = inputs_desc_;
        nb->outputs_desc_ = outputs_desc_;
        nb->misc_info_ = misc_info_;
        nb->initialized_ = initialized_;
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

//
// Created by aichao on 2025/8/2.
// Sophgo 算能 TPU 推理后端：libsophon bmrt 直接推理（替代 sail）。
// 用 bmrt 是为了支持设备内存（bm_device_mem_t）输入零拷贝推理（infer_device），
// 配合 SophgoProcessorBackend 的 BMCV 设备预处理，跳过 CPU 往返拷贝。
//

#include "core/md_log.h"
#include "runtime/backends/sophgo/sophgo_backend.h"

#ifdef ENABLE_SOPHGO
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

    SophgoBackend::~SophgoBackend() {
#ifdef ENABLE_SOPHGO
        // 缓存的 input/output 设备内存由 bmrt_tensor 分配，bmrt_destroy 统一释放。
        // 这里手动 bm_free_device_mem 会导致 double-free 段错误（与官方 SOPHON-DEMO
        // 只调 bmrt_destroy + bm_dev_free 的行为一致）。
        if (cached_in_mems_) {
            delete[] static_cast<bm_device_mem_t*>(cached_in_mems_);
            cached_in_mems_ = nullptr;
        }
        if (cached_out_mems_) {
            delete[] static_cast<bm_device_mem_t*>(cached_out_mems_);
            cached_out_mems_ = nullptr;
        }
        io_cached_ = false;
        if (bmrt_) {
            bmrt_destroy(static_cast<void*>(bmrt_));
            bmrt_ = nullptr;
        }
        if (handle_) {
            bm_dev_free(static_cast<bm_handle_t>(handle_));
            handle_ = nullptr;
        }
        net_info_ = nullptr;
#endif
    }

    bool SophgoBackend::init(const RuntimeOption& option) {
        if (initialized_) {
            MD_LOG_ERROR << "SophgoBackend is already initialized." << std::endl;
            return false;
        }
        bmodel_path_ = option.sophgo_option.bmodel_path.empty()
            ? option.model_file : option.sophgo_option.bmodel_path;
        const int device_id = option.sophgo_option.device_id;

        bm_handle_t h = nullptr;
        if (bm_dev_request(&h, device_id) != BM_SUCCESS) {
            MD_LOG_ERROR << "[SophgoBackend] bm_dev_request failed (device " << device_id << ")." << std::endl;
            return false;
        }
        handle_ = static_cast<void*>(h);
        static bm_misc_info m{};
        bm_get_misc_info(h, &m);
        misc_info_ = &m;

        void* bmrt = bmrt_create(h);
        if (!bmrt) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_create failed." << std::endl;
            bm_dev_free(h);
            handle_ = nullptr;
            return false;
        }
        if (!bmrt_load_bmodel(bmrt, bmodel_path_.c_str())) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_load_bmodel failed: " << bmodel_path_ << std::endl;
            bmrt_destroy(bmrt);
            bm_dev_free(h);
            handle_ = nullptr;
            return false;
        }
        bmrt_ = bmrt;

        const char** net_names = nullptr;
        bmrt_get_network_names(bmrt, &net_names);
        if (!net_names || !net_names[0]) {
            MD_LOG_ERROR << "[SophgoBackend] no network in bmodel." << std::endl;
            return false;
        }
        graph_name_ = net_names[0];
        const bm_net_info_t* info = bmrt_get_network_info(bmrt, graph_name_.c_str());
        if (!info) {
            MD_LOG_ERROR << "[SophgoBackend] get_network_info failed: " << graph_name_ << std::endl;
            return false;
        }
        net_info_ = info;

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

        initialized_ = true;
        MD_LOG_INFO << "SophgoBackend(bmrt) loaded " << bmodel_path_
            << " graph[" << graph_name_ << "] inputs=" << inputs_desc_.size()
            << " outputs=" << outputs_desc_.size() << std::endl;
        return true;
    }

    bool SophgoBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (!initialized_ || !bmrt_ || !net_info_) return false;
        bm_handle_t h = static_cast<bm_handle_t>(handle_);
        if (inputs.size() != inputs_desc_.size()) {
            MD_LOG_ERROR << "[SophgoBackend] inputs size mismatch: " << inputs.size()
                << " vs " << inputs_desc_.size() << std::endl;
            return false;
        }
        const bm_net_info_t* info = static_cast<const bm_net_info_t*>(net_info_);
        const size_t ni = inputs.size();
        const size_t no = outputs_desc_.size();

        std::vector<bm_tensor_t> in_t(ni), out_t(no);
        // 缓存 io 设备内存（首次 bmrt_tensor 分配并缓存，之后复用，避免每次 free 的稳定性问题）
        if (!io_cached_ || !cached_in_mems_ || !cached_out_mems_) {
            if (!ensure_io_cache()) return false;
        }
        bm_device_mem_t* ins = static_cast<bm_device_mem_t*>(cached_in_mems_);
        bm_device_mem_t* outs = static_cast<bm_device_mem_t*>(cached_out_mems_);
        for (size_t i = 0; i < ni; ++i) {
            in_t[i].device_mem = ins[i];
            in_t[i].dtype = info->input_dtypes[i];
            in_t[i].shape = info->stages[0].input_shapes[i];
        }
        for (size_t i = 0; i < no; ++i) {
            out_t[i].device_mem = outs[i];
            out_t[i].dtype = info->output_dtypes[i];
            out_t[i].shape = info->stages[0].output_shapes[i];
        }

        for (size_t i = 0; i < ni; ++i) {
            if (bm_memcpy_s2d(h, in_t[i].device_mem, inputs[i].data()) != BM_SUCCESS) {
                MD_LOG_ERROR << "[SophgoBackend] bm_memcpy_s2d(input) failed." << std::endl;
                return false;
            }
        }

        if (!bmrt_launch_tensor_ex(bmrt_, graph_name_.c_str(), in_t.data(), static_cast<int>(ni),
                                   out_t.data(), static_cast<int>(no),
                                   /*user_mem*/true, /*user_stmode*/false)) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_launch_tensor failed." << std::endl;
            return false;
        }
        bm_thread_sync(h);

        outputs->resize(no);
        for (size_t i = 0; i < no; ++i) {
            // 输出 shape/dtype 以 bmodel 静态信息为准（bmrt_launch_tensor 可能改写 out_t[i]）
            const auto os = info->stages[0].output_shapes[i];
            std::vector<int64_t> shape64(os.dims, os.dims + os.num_dims);
            outputs_desc_[i].dtype = bm_dtype_to_md(info->output_dtypes[i]);
            (*outputs)[i].allocate(shape64, outputs_desc_[i].dtype, Device::CPU, outputs_desc_[i].name);
            if (bm_memcpy_d2s(h, (*outputs)[i].data(), out_t[i].device_mem) != BM_SUCCESS) {
                MD_LOG_ERROR << "[SophgoBackend] bm_memcpy_d2s(output) failed." << std::endl;
                return false;
            }
        }
        return true;
    }

    bool SophgoBackend::infer_device(void* device_input, const std::vector<int64_t>& shape,
                                     std::vector<Tensor>* outputs) {
        if (!initialized_ || !bmrt_ || !net_info_ || !device_input) return false;
        bm_handle_t h = static_cast<bm_handle_t>(handle_);
        const bm_net_info_t* info = static_cast<const bm_net_info_t*>(net_info_);
        const size_t no = outputs_desc_.size();

        // 零拷贝输入：device_input 为 BMCV 预处理 attach 写入的输入设备内存（bmrt_tensor 分配）
        bm_shape_t shp;
        std::vector<int> dims_int(shape.begin(), shape.end());
        bmrt_shape(&shp, dims_int.data(), static_cast<int>(dims_int.size()));
        bm_tensor_t in_t;
        bmrt_tensor_with_device(&in_t, *static_cast<bm_device_mem_t*>(device_input),
                                info->input_dtypes[0], shp);

        std::vector<bm_tensor_t> out_t(no);
        // 使用缓存的输出设备内存（避免每帧 bm_free_device_mem 导致 bmrt 状态异常/段错误）
        if (!io_cached_ || !cached_out_mems_) {
            if (!ensure_io_cache()) return false;
        }
        bm_device_mem_t* outs = static_cast<bm_device_mem_t*>(cached_out_mems_);
        for (size_t i = 0; i < no; ++i) {
            out_t[i].device_mem = outs[i];
            out_t[i].dtype = info->output_dtypes[i];
            out_t[i].shape = info->stages[0].output_shapes[i];
        }
        if (!bmrt_launch_tensor_ex(bmrt_, graph_name_.c_str(), &in_t, 1,
                                   out_t.data(), static_cast<int>(no),
                                   /*user_mem*/true, /*user_stmode*/false)) {
            MD_LOG_ERROR << "[SophgoBackend] bmrt_launch_tensor(device) failed." << std::endl;
            return false;
        }
        bm_thread_sync(h);

        outputs->resize(no);
        // SOC 模式用 mmap 零拷贝读输出，PCIe 用 d2s 拷贝
        const bm_misc_info* mi = static_cast<const bm_misc_info*>(misc_info_);
        const bool is_soc = mi && mi->pcie_soc_mode == 1;
        for (size_t i = 0; i < no; ++i) {
            // 输出 shape/dtype 以 bmodel 静态信息为准（bmrt_launch_tensor 会改写 out_t 的 shape 为非法值）
            const auto os = info->stages[0].output_shapes[i];
            const std::vector<int64_t> shape64(os.dims, os.dims + os.num_dims);
            outputs_desc_[i].dtype = bm_dtype_to_md(info->output_dtypes[i]);
            (*outputs)[i].allocate(shape64, outputs_desc_[i].dtype, Device::CPU, outputs_desc_[i].name);
            if (is_soc && out_t[i].dtype == BM_FLOAT32) {
                unsigned long long addr = 0;
                if (bm_mem_mmap_device_mem(h, &out_t[i].device_mem, &addr) != BM_SUCCESS ||
                    bm_mem_invalidate_device_mem(h, &out_t[i].device_mem) != BM_SUCCESS) {
                    MD_LOG_ERROR << "[SophgoBackend] mmap output failed." << std::endl;
                    return false;
                }
                memcpy((*outputs)[i].data(), reinterpret_cast<void*>(addr),
                       static_cast<size_t>(bm_mem_get_device_size(out_t[i].device_mem)));
                bm_mem_unmap_device_mem(h, reinterpret_cast<void*>(addr),
                                        bm_mem_get_device_size(out_t[i].device_mem));
            } else {
                if (bm_memcpy_d2s(h, (*outputs)[i].data(), out_t[i].device_mem) != BM_SUCCESS) {
                    MD_LOG_ERROR << "[SophgoBackend] bm_memcpy_d2s(output) failed." << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    void* SophgoBackend::get_input_device_mem() {
        if (!initialized_ || !bmrt_ || !net_info_) return nullptr;
        if (!io_cached_ || !cached_in_mems_) {
            if (!ensure_io_cache()) return nullptr;
        }
        return cached_in_mems_;
    }

    void* SophgoBackend::get_output_device_mem() {
        if (!initialized_ || !bmrt_ || !net_info_) return nullptr;
        if (!io_cached_ || !cached_out_mems_) {
            if (!ensure_io_cache()) return nullptr;
        }
        return cached_out_mems_;
    }

    bool SophgoBackend::ensure_io_cache() {
        if (io_cached_ && cached_in_mems_ && cached_out_mems_) return true;
        if (!bmrt_ || !net_info_) return false;
        const bm_net_info_t* info = static_cast<const bm_net_info_t*>(net_info_);
        const size_t ni = inputs_desc_.size();
        const size_t no = outputs_desc_.size();
        auto* ins = new bm_device_mem_t[ni];
        for (size_t i = 0; i < ni; ++i) {
            bm_tensor_t t;
            if (!bmrt_tensor(&t, bmrt_, info->input_dtypes[i],
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
            if (!bmrt_tensor(&t, bmrt_, info->output_dtypes[i],
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

    void* SophgoBackend::get_bm_handle() {
        return handle_;
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
    bool SophgoBackend::infer_device(void*, const std::vector<int64_t>&, std::vector<Tensor>*) { return false; }
    void* SophgoBackend::get_input_device_mem() { return nullptr; }
    void* SophgoBackend::get_output_device_mem() { return nullptr; }
    void* SophgoBackend::get_bm_handle() { return nullptr; }
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

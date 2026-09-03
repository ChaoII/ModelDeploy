#include <filesystem>
#include <fstream>
#include <algorithm>
#include <ncnn/net.h>
#include "core/md_log.h"
#include "runtime/backends/ncnn/ncnn_backend.h"

namespace modeldeploy {

    namespace {
        std::string replace_ext(const std::string& path, const char* ns) {
            auto p = std::filesystem::path(path);
            return p.replace_extension(ns).string();
        }

        std::vector<int64_t> mat_shape(const ncnn::Mat& m) {
            if (m.dims == 1) return {static_cast<int64_t>(m.w)};
            if (m.dims == 2) return {static_cast<int64_t>(m.h), static_cast<int64_t>(m.w)};
            if (m.dims == 3) return {static_cast<int64_t>(m.c), static_cast<int64_t>(m.h), static_cast<int64_t>(m.w)};
            // ncnn dims==4 的 Mat 布局为 (w,h,d,c)；此处将 (w,h,d,c) 映射为 SDK 的 (c,d,h,w)。
            // 注：该 dims==4 输出映射 {c,d,h,w} 待 seg/sem/depth（4D 输出）接入时复核（Phase A Task 3-4）；
            // 当前 batch=1 图像输入（[1,c,h,w]）实际产生 dims==3 输出，走上一分支。
            return {static_cast<int64_t>(m.c), static_cast<int64_t>(m.d), static_cast<int64_t>(m.h), static_cast<int64_t>(m.w)};
        }

        size_t mat_total(const ncnn::Mat& m) {
            return static_cast<size_t>(m.w) * m.h * (m.dims >= 3 ? m.c : 1) * (m.dims >= 4 ? m.d : 1);
        }
    }

    bool NcnnBackend::init(const RuntimeOption& runtime_option) {
        const_cast<RuntimeOption&>(runtime_option).validate();
        if (initialized_) {
            MD_LOG_ERROR << "NcnnBackend is already initialized, cannot initialize again." << std::endl;
            return false;
        }
        saved_option_ = runtime_option;
        option_ = runtime_option.ncnn_option;
        net_ = std::make_unique<ncnn::Net>();
        net_->opt.num_threads = option_.cpu_thread_num > 0 ? option_.cpu_thread_num : 4;
        bool vulkan = (runtime_option.device == Device::VULKAN);
        net_->opt.use_vulkan_compute = vulkan;
        if (vulkan) {
            net_->set_vulkan_device(runtime_option.device_id >= 0 ? runtime_option.device_id : option_.device_id);
        }

        if (option_.model_from_memory) {
            size_t pb = net_->load_param(reinterpret_cast<const unsigned char*>(option_.param_buffer.data()));
            if (pb == 0) { MD_LOG_ERROR << "ncnn load_param(mem) failed." << std::endl; return false; }
            size_t mb = net_->load_model(reinterpret_cast<const unsigned char*>(option_.bin_buffer.data()));
            if (mb == 0) { MD_LOG_ERROR << "ncnn load_model(mem) failed." << std::endl; return false; }
        } else {
            std::string bin_path = replace_ext(runtime_option.model_file, ".bin");
            if (net_->load_param(runtime_option.model_file.c_str()) != 0) {
                MD_LOG_ERROR << "ncnn load_param failed: " << runtime_option.model_file << std::endl;
                return false;
            }
            if (net_->load_model(bin_path.c_str()) != 0) {
                MD_LOG_ERROR << "ncnn load_model failed: " << bin_path << std::endl;
                return false;
            }
        }

        for (auto* n : net_->input_names()) input_names_.emplace_back(n);
        for (auto* n : net_->output_names()) output_names_.emplace_back(n);

        MD_LOG_INFO << "ncnn loaded " << input_names_.size() << " input(s), "
                    << output_names_.size() << " output(s), device="
                    << (vulkan ? "VULKAN" : "CPU") << "." << std::endl;
        initialized_ = true;
        return true;
    }

    TensorInfo NcnnBackend::get_input_info(int index) {
        if (index < 0 || index >= static_cast<int>(num_inputs())) {
            MD_LOG_FATAL << "input index " << index << " out of range." << std::endl;
        }
        TensorInfo info;
        info.name = input_names_[index];
        info.shape = {-1};
        info.dtype = DataType::FP32;
        return info;
    }

    TensorInfo NcnnBackend::get_output_info(int index) {
        if (index < 0 || index >= static_cast<int>(num_outputs())) {
            MD_LOG_FATAL << "output index " << index << " out of range." << std::endl;
        }
        TensorInfo info;
        info.name = output_names_[index];
        info.shape = {-1};
        info.dtype = DataType::FP32;
        return info;
    }

    std::vector<TensorInfo> NcnnBackend::get_input_infos() {
        std::vector<TensorInfo> v;
        for (size_t i = 0; i < num_inputs(); ++i) v.push_back(get_input_info(static_cast<int>(i)));
        return v;
    }

    std::vector<TensorInfo> NcnnBackend::get_output_infos() {
        std::vector<TensorInfo> v;
        for (size_t i = 0; i < num_outputs(); ++i) v.push_back(get_output_info(static_cast<int>(i)));
        return v;
    }

    bool NcnnBackend::infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) {
        if (inputs.size() != num_inputs() || !outputs) {
            MD_LOG_ERROR << "input/output count mismatch." << std::endl;
            return false;
        }
        ncnn::Extractor ex = net_->create_extractor();
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& t = inputs[i];
            if (t.dtype() != DataType::FP32) { MD_LOG_ERROR << "ncnn requires FP32 input." << std::endl; return false; }
            auto shp = t.shape();
            if (shp.size() == 4) {
                // ncnn 4D 输入的 Mat 构造参数为 (w,h,c)（dims==4 时实际是 (w,h,d,c)）；SDK 输入 Tensor 顺序为 (n,c,d,h,w)。
                // 当前按 batch=1 语义处理（假设 shp[0]==1），使 batch 维与深度维 c/d 对称、全链路一致。
                // 注意：batch=1 图像输入（如 [1,3,64,64]）在此被压掉 batch 维，以 Mat(w=64,h=64,c=3) 喂 3D 卷积，
                // 对应输出即 CHW dims==3。batch>1 仍需显式扩充（未实现）。
                if (shp[0] != 1) {
                    MD_LOG_WARN << "ncnn 4D 输入按 batch=1 语义处理，忽略 batch=" << shp[0] << std::endl;
                }
                ncnn::Mat in_mat(static_cast<int>(shp[3]), static_cast<int>(shp[2]),
                                 static_cast<int>(shp[1]) /* w,h,c */);
                memcpy(in_mat.data, t.data(), t.byte_size());
                ex.input(input_names_[i].c_str(), in_mat);
            } else if (shp.size() == 3) {
                ncnn::Mat in_mat(static_cast<int>(shp[2]), static_cast<int>(shp[1]), static_cast<int>(shp[0]));
                memcpy(in_mat.data, t.data(), t.byte_size());
                ex.input(input_names_[i].c_str(), in_mat);
            } else {
                MD_LOG_ERROR << "unsupported input dims for ncnn." << std::endl;
                return false;
            }
        }
        outputs->resize(num_outputs());
        for (size_t i = 0; i < num_outputs(); ++i) {
            ncnn::Mat out_mat;
            if (ex.extract(output_names_[i].c_str(), out_mat) != 0) {
                MD_LOG_ERROR << "ncnn extract failed: " << output_names_[i] << std::endl;
                return false;
            }
            ncnn::Mat plain;
            if (out_mat.elempack != 1) {
                ncnn::convert_packing(out_mat, plain, 1);
            } else {
                plain = out_mat;
            }
            size_t total = mat_total(plain);
            auto sh = mat_shape(plain);
            (*outputs)[i].allocate(sh, DataType::FP32, Device::CPU, output_names_[i]);
            if ((*outputs)[i].byte_size() / sizeof(float) != total) {
                MD_LOG_ERROR << "output size mismatch." << std::endl;
                return false;
            }
            memcpy((*outputs)[i].data(), plain.data, (*outputs)[i].byte_size());
        }
        return true;
    }

    std::unique_ptr<BaseBackend> NcnnBackend::clone(const RuntimeOption&, void*, int) {
        if (!net_) return nullptr;
        auto nb = std::make_unique<NcnnBackend>();
        if (!nb->init(saved_option_)) {
            MD_LOG_ERROR << "NcnnBackend clone: re-init failed." << std::endl;
            return nullptr;
        }
        return nb;
    }

} // namespace modeldeploy

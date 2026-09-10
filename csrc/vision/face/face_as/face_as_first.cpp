//
// Created by aichao on 2025/3/26.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_as/face_as_first.h"
#include <cstring>


namespace modeldeploy::vision::face {
    SeetaFaceAsFirst::SeetaFaceAsFirst(const std::string& model_file,
                                       const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool SeetaFaceAsFirst::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        backend_ = create_processor_backend(runtime_option.device, runtime_option.backend,
                                            runtime_option.device_id);
        return true;
    }

    bool SeetaFaceAsFirst::preprocess(ImageData* image, Tensor* output) {
        // 输入为对齐后的 BGR 图（固定 256x256，可能其它尺寸）。
        // CenterCrop(224) + BGR2YCrCb + cast(float) + HWC2CHW 全融合为单步 SIMD kernel。
        const int src_w = image->width();
        // 非 256 时先 resize 到 256 再 center_crop 224（与原始语义一致，合并为单步映射）。
        // fused 映射：src = (dst - origin)/scale。
        //   resize 到 256：scale_resize = src_w/256（dst 256 像素覆盖 src 全部）
        //   center_crop 224 from 256：origin_crop = -16（dst=0 -> src=16）
        // 合并：origin = -16 / (src_w/256) * ... 简化直接按 face_rec 范式：
        //   scale = 256/src_w, origin = -16.0f * scale? 验证：dst=0 -> src=(0-origin)/scale
        // 采用与 face_rec 一致推导：scale = 256/src_w, origin = -16
        //   src = (dst - origin)/scale = (dst+16)/(256/src_w)
        //   256 输入：src = dst+16（crop [16,240)）✓
        //   512 输入：src = 2*(dst+16)（先缩到 256 再 crop，等效）✓
        const float scale = 256.0f / static_cast<float>(src_w);
        const float origin = -16.0f;
        const float ox = origin, oy = origin;
        const float sx = scale, sy = scale;

        // BT.601 YCrCb（对采样出的 r,g,b，展开为 3x3 矩阵）。
        // 系数来自 OpenCV BGR2YCrCb 反标定（BT.601，float 精度）：
        //   Y  = 0.299r + 0.587g + 0.114b
        //   Cr = 0.500r - 0.419g - 0.081b + 128
        //   Cb = -0.169r - 0.331g + 0.500b + 128
        const float mat[3][3] = {
            { 0.299f, 0.587f, 0.114f },
            { 0.499744f, -0.418548f, -0.081265f },
            { -0.168664f, -0.331215f, 0.499916f },
        };
        const float bias[3] = { 0.0f, 128.0165f, 127.9938f };

        if (!backend_->fused_preprocess_color_matrix(*image, output, size_,
                                                     ox, oy, sx, sy, mat, bias, 0.0f)) {
            MD_LOG_ERROR << "Failed to fused color matrix preprocess." << std::endl;
            return false;
        }
        return true;
    }


    bool SeetaFaceAsFirst::postprocess(const std::vector<Tensor>& infer_result, float* result) {
        const Tensor& infer_result_ = infer_result[0];
        *result = static_cast<const float*>(infer_result_.data())[1];
        return true;
    }

    bool SeetaFaceAsFirst::predict(const ImageData& image, float* result) {
        std::vector<Tensor> input_tensors(1);
        auto _image = image;
        if (!preprocess(&_image, &input_tensors[0])) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].set_name(get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        postprocess(output_tensors, result);
        return true;
    }

    bool SeetaFaceAsFirst::batch_predict(const std::vector<ImageData>& images,
                                         std::vector<float>* results) {
        if (images.empty() || results == nullptr) return false;
        const int n = static_cast<int>(images.size());
        results->resize(n);
        if (n == 1) {
            return predict(images[0], &(*results)[0]);
        }
        // 每图独立 fused 预处理到临时 tensor，再合并为一个 batch tensor 一次推理
        // （fas_first.onnx 输入全动态 [-1,-1,-1,-1]，支持动态 batch）
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        const int plane = 3 * dst_h * dst_w;
        Tensor batch_tensor({n, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* batch_data = batch_tensor.data_ptr<float>();
        for (int i = 0; i < n; ++i) {
            Tensor single;
            auto img = images[i];
            if (!preprocess(&img, &single)) {
                MD_LOG_ERROR << "Failed to preprocess input image " << i << "." << std::endl;
                return false;
            }
            std::memcpy(batch_data + static_cast<size_t>(i) * plane, single.data(), plane * sizeof(float));
        }
        batch_tensor.set_name(get_input_info(0).name);
        std::vector<Tensor> input_tensors{batch_tensor};
        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to batch inference." << std::endl;
            return false;
        }
        // postprocess：输出 [N, C]，每行取 [1]（与单图 postprocess 一致）
        const float* out_data = static_cast<const float*>(output_tensors[0].data());
        const int num_classes = static_cast<int>(output_tensors[0].size()) / n;
        for (int i = 0; i < n; ++i) {
            (*results)[i] = out_data[static_cast<size_t>(i) * num_classes + 1];
        }
        return true;
    }

    std::unique_ptr<SeetaFaceAsFirst> SeetaFaceAsFirst::clone() const {
        auto clone_model = std::make_unique<SeetaFaceAsFirst>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
}


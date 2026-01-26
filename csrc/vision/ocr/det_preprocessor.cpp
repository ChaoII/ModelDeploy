//
// Created by aichao on 2025/2/21.
//

#include "vision/utils.h"
#include "vision/ocr/det_preprocessor.h"
#ifdef WITH_GPU
#include "vision/common/processors/fusion_resize_pad_normalize_permute.cuh"
#endif
#include <vision/common/processors/fusion_resize_pad_normalize_permute.h>
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/common/processors/resize.h"

namespace modeldeploy::vision::ocr {
    std::array<int, 4> DBDetectorPreprocessor::ocr_detector_get_info(
        const ImageData* image, const int max_size_len) const {
        const int w = image->width();
        const int h = image->height();
        if (static_img_size_.size() == 2) {
            return {w, h, static_img_size_[0], static_img_size_[1]};
        }
        float ratio = 1.0f;
        const int max_wh = std::max(w, h);
        if (max_wh > max_size_len) {
            ratio = static_cast<float>(max_size_len) / max_wh;
        }
        // 最小图片尺寸为32
        int resize_h = std::max(static_cast<int>(h * ratio), 32);
        int resize_w = std::max(static_cast<int>(w * ratio), 32);
        // 如果尺寸不是32的倍数，则取最接近的32的倍数向上取整
        resize_h = (resize_h + 31) / 32 * 32;
        resize_w = (resize_w + 31) / 32 * 32;
        return {w, h, resize_w, resize_h};
    }

    DBDetectorPreprocessor::DBDetectorPreprocessor() {
    }

    bool DBDetectorPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                            const std::vector<int>& resize_size,
                                            const std::vector<int>& dst_size) const {
        if (use_cuda_preproc_) {
#ifdef WITH_GPU

            return fusion_resize_pad_normalize_permute_cuda(image, output,
                                                            resize_size,
                                                            dst_size,
                                                            mean_,
                                                            std_,
                                                            pad_value_);
#else
            MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, rollback to cpu" << std::endl;
#endif
        }
        return fusion_resize_pad_normalize_permute_cpu(image, output,
                                                       resize_size,
                                                       dst_size,
                                                       mean_,
                                                       std_,
                                                       pad_value_);
    }


    bool DBDetectorPreprocessor::apply(const std::vector<ImageData>& image_batch,
                                       std::vector<Tensor>* outputs) {
        // 组batch找到当前batch最大的宽高组织
        const size_t batch_size = image_batch.size();
        batch_info_.resize(batch_size);
        int max_resize_w = 0;
        int max_resize_h = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            batch_info_[i] = ocr_detector_get_info(&image_batch.at(i), max_side_len_);
            max_resize_w = std::max(max_resize_w, batch_info_[i][2]);
            max_resize_h = std::max(max_resize_h, batch_info_[i][3]);
        }
        outputs->resize(1);
        std::vector<Tensor> tensors(image_batch.size());
        for (size_t i = 0; i < batch_size; ++i) {
            ImageData image = image_batch.at(i);
            // resize 到指定的32倍数的尺寸，然后pad到max_size,fuse_resize_and_pad和letterbox差别很大
            preprocess(image, &tensors[i], {batch_info_[i][2], batch_info_[i][3]}, {max_resize_w, max_resize_h});
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}

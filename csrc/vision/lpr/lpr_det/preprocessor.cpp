//
// Created by aichao on 2025/6/10.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/lpr/lpr_det/preprocessor.h"

namespace modeldeploy::vision::lpr {
    LprDetPreprocessor::LprDetPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }


    bool LprDetPreprocessor::preprocess(const ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const {
        // yolov8's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        const int src_w = image->width();
        const int src_h = image->height();
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, size_);
        float origin_x, origin_y, scale_x, scale_y;
        utils::letter_box_to_fused_params(*letter_box_record,
                                          &origin_x, &origin_y, &scale_x, &scale_y);
        const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
        // pad 114 在仿射后空间（归一化）：114/255
        const float pad_norm = padding_value_[0] / 255.0f;
        if (!backend_->fused_preprocess(*image, output, size_,
                                        origin_x, origin_y, scale_x, scale_y,
                                        alpha, beta, true, pad_norm)) return false;
        return true;
    }

    bool LprDetPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images->size());
        outputs->resize(1);
        if (images->size() == 1) {
            // 单图：直接写进持久 outputs[0]，其 allocate 跨帧复用显存 buffer
            preprocess(&(*images)[0], &(*outputs)[0], &(*letter_box_records)[0]);
            return true;
        }
        // 整批一次 fused kernel（每图独立 letterbox 映射）
        const int n = static_cast<int>(images->size());
        std::vector<float> oxs(n), oys(n), sxs(n), sys(n);
        for (int i = 0; i < n; ++i) {
            (*letter_box_records)[i] = utils::cal_letter_box_param(
                {(*images)[i].width(), (*images)[i].height()}, size_);
            utils::letter_box_to_fused_params((*letter_box_records)[i],
                                              &oxs[i], &oys[i], &sxs[i], &sys[i]);
        }
        const float pad_norm = padding_value_[0] / 255.0f;
        if (!backend_->fused_preprocess_batch(*images, &(*outputs)[0], size_,
                                              oxs, oys, sxs, sys,
                                              {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f},
                                              {0.0f, 0.0f, 0.0f}, true, pad_norm)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        return true;
    }
}

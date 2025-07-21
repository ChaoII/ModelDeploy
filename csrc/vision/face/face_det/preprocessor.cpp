//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_det/preprocessor.h"
#include <vision/common/processors/cast.h>
#include <vision/common/processors/color_space_convert.h>
#include <vision/common/processors/convert.h>
#include <vision/common/processors/hwc2chw.h>

namespace modeldeploy::vision::face {
    ScrfdPreprocessor::ScrfdPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {0.0, 0.0, 0.0};
        is_mini_pad_ = false;
        is_no_pad_ = false;
        is_scale_up_ = true;
        stride_ = 32;
    }


    bool ScrfdPreprocessor::preprocess(ImageData* image, Tensor* output, LetterBoxRecord* letter_box_record) const {
        // scrfd's preprocess steps
        // 1. letterbox
        // 2. BGR->RGB
        // 3. HWC->CHW
        cv::Mat mat;
        image->to_mat(&mat);
        utils::letter_box(&mat, size_, is_scale_up_, is_mini_pad_, is_no_pad_, padding_value_, stride_,
                          letter_box_record);
        BGR2RGB::apply(&mat);
        // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
        //                std::vector<float>(mat->Channels(), 1.0));
        // Compute `result = mat * alpha + beta` directly by channel
        // Original Repo/tools/scrfd.py: cv2.dnn.blobFromImage(img, 1.0/128,
        // input_size, (127.5, 127.5, 127.5), swapRB=True)
        const std::vector alpha = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
        const std::vector beta = {-127.5f / 128.f, -127.5f / 128.f, -127.5f / 128.f};
        Convert::apply(&mat, alpha, beta);
        HWC2CHW::apply(&mat);
        Cast::apply(image, "float");
        if (!utils::mat_to_tensor(mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool ScrfdPreprocessor::run(
        std::vector<ImageData>* images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images->empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(1);
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images->size());
        for (size_t i = 0; i < images->size(); ++i) {
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
            preprocess(&(*images)[i], &tensors[i], &(*letter_box_records)[i]);
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

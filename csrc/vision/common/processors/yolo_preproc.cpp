//
// Created by aichao on 2025/7/22.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/convert_and_permute.h"


namespace modeldeploy::vision {
    bool yolo_preprocess_cpu(const ImageData* image, Tensor* output,
                             const std::vector<int>& dst_size,
                             const std::vector<float>& pad_val,
                             LetterBoxRecord* letter_box_record) {
        // yolo's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        cv::Mat mat;
        image->to_mat(&mat);
        utils::letter_box(&mat, dst_size, pad_val, letter_box_record);
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        ConvertAndPermute::apply(&mat, alpha, beta, true);
        utils::mat_to_tensor(mat, output, false);
        output->expand_dim(0); // reshape to n, c, h, w
        // 这里不能深拷贝，深拷贝cv::Mat mat持有的内存就会被释放那么output指向的内存也会被释放
        // image->update_from_mat(&mat);
        // *image = ImageData::from_mat(&mat);
        return true;
    }
}

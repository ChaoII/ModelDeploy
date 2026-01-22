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
        *letter_box_record = utils::cal_letter_box_param({image->width(), image->height()}, dst_size);
        auto s = image->letter_box(dst_size, 114.0f).fuse_convert_and_permute();
        // s = s.normalize({1/255.0f, 1/255.0f, 1/255.0f}, {0.0f, 0.0f, 0.0f});
        // s.imshow("letter_box");
        // s = s.permute();
        s.to_tensor(output);
        output->expand_dim(0); // reshape to n, c, h, w
        // 这里不能深拷贝，深拷贝cv::Mat mat持有的内存就会被释放那么output指向的内存也会被释放
        // image->update_from_mat(&mat);
        // *image = ImageData::from_mat(&mat);
        return true;
    }
}

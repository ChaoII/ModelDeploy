//
// Created by aichao on 2025/5/30.
//

#include <numeric>
#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/obb/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsObbPostprocessor::UltralyticsObbPostprocessor() {
        conf_threshold_ = 0.25;
        mask_threshold_ = 0.35;
        nms_threshold_ = 0.5;
    }

    bool UltralyticsObbPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<DetectionResult>* results,
        const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,84,8400)->(1,8400,84) 84 = 4(xc,yc,w,h)+80(coco 80 classes)
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 1}).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            (*results)[bs].clear();
            (*results)[bs].reserve(static_cast<int>(tensor_transpose.shape()[1]));
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            // 官方模型为21504
            const auto ooj_num = tensor_transpose.shape()[1];
            // 官方模型为4(xc, yc, w, h)+classes_num(15)+1(angle) = 20
            const auto obj_attr_num = tensor_transpose.shape()[2];
            const float* data = static_cast<const float*>(tensor_transpose.data()) + bs * ooj_num * obj_attr_num;
            for (size_t i = 0; i < ooj_num; ++i) {
                const auto attr_ptr = data + i * obj_attr_num;
                // 4(xc, yc, w, h)+classes_num(15)+1(angle)
                const float* max_class_score = std::max_element(attr_ptr + 4,
                                                                attr_ptr + obj_attr_num - 1);
                float confidence = *max_class_score;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(std::distance(attr_ptr + 4, max_class_score));
                // convert from [x, y, w, h, a] to [x1, y1, x2, y2,x3, y3, x4, y4]
                // 其中a为angle矩形框的旋转角度, 默认为弧度制
                (*results)[bs].rotated_boxes.emplace_back(std::array{
                    utils::xcycwha_to_x1y1x2y2x3y3x4y4(
                        attr_ptr[0],
                        attr_ptr[1],
                        attr_ptr[2],
                        attr_ptr[3],
                        attr_ptr[obj_attr_num - 1])
                });
                (*results)[bs].label_ids.push_back(label_id);
                (*results)[bs].scores.push_back(confidence);
            }
            if ((*results)[bs].rotated_boxes.empty()) {
                continue;
            }
            utils::obb_nms(&(*results)[bs], nms_threshold_);
        }
        return true;
    }
}

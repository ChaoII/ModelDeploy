//
// Created by aichao on 2025/06/2.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/pose/postprocessor.h"

namespace modeldeploy::vision::detection
{
    UltralyticsPosePostprocessor::UltralyticsPosePostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
        keypoints_num_ = 17;
    }

    bool UltralyticsPosePostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<PoseResult>* results,
        const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,84,8400)->(1,8400,84) 84 = 4(xc,yc,w,h)+80(coco 80 classes)
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 1}).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            (*results)[bs].clear();
            (*results)[bs].reserve(tensor_transpose.shape()[1]);
            (*results)[bs].keypoints_per_instance = keypoints_num_;
            int64_t keypoints_num_by_output = (tensor_transpose.shape()[2] - 5) / 3;
            if (keypoints_num_by_output != keypoints_num_) {
                MD_LOG_ERROR << "Keypoints num set error, set" << keypoints_num_ << "but is: " <<
                    keypoints_num_by_output << "." << std::endl;
            }
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const float* data =
                static_cast<const float*>(tensor_transpose.data()) +
                bs * tensor_transpose.shape()[1] * tensor_transpose.shape()[2];
            for (size_t i = 0; i < tensor_transpose.shape()[1]; ++i) {
                // 4(xc,yc,w,h)+1(conf)+17(keypoints)*3(x,y,conf)=56
                const float* attr_ptr = data + i * tensor_transpose.shape()[2];
                float confidence = attr_ptr[4];
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                int32_t label_id = 0;
                // convert from [xc, yc, w, h] to [x, y, width, height]
                (*results)[bs].boxes.emplace_back(
                    attr_ptr[0] - attr_ptr[2] / 2.0f,
                    attr_ptr[1] - attr_ptr[3] / 2.0f,
                    attr_ptr[2],
                    attr_ptr[3]
                );

                (*results)[bs].label_ids.push_back(label_id);
                (*results)[bs].scores.push_back(confidence);

                if (keypoints_num_ > 0) {
                    const float* keypoints_ptr = attr_ptr + 5;
                    for (size_t j = 0; j < keypoints_num_ * 3; j += 3) {
                        (*results)[bs].keypoints.emplace_back(
                            keypoints_ptr[j], keypoints_ptr[j + 1], keypoints_ptr[j + 2]);
                    }
                }
            }

            if ((*results)[bs].boxes.empty()) {
                continue;
            }
            utils::nms(&(*results)[bs], nms_threshold_);
            // scale the boxes to the origin image shape
            auto iter_out = ims_info[bs].find("output_shape");
            auto iter_ipt = ims_info[bs].find("input_shape");
            if (!(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end())) {
                MD_LOG_ERROR << "Cannot find input_shape or output_shape from im_info." << std::endl;
            }
            const float out_h = iter_out->second[0];
            const float out_w = iter_out->second[1];
            const float ipt_h = iter_ipt->second[0];
            const float ipt_w = iter_ipt->second[1];
            const float scale = std::min(out_h / ipt_h, out_w / ipt_w);
            const float pad_h = (out_h - ipt_h * scale) / 2;
            const float pad_w = (out_w - ipt_w * scale) / 2;

            for (auto& keypoint : (*results)[bs].keypoints) {
                keypoint.x = (keypoint.x - pad_w) / scale;
                keypoint.y = (keypoint.y - pad_h) / scale;
            }

            for (auto& box : (*results)[bs].boxes) {
                // clip box()
                //先减去 padding;
                //再除以缩放因子 scale;
                //最后限制在原始图像范围内 [0, width], [0, height]。
                // 去掉 padding 并缩放
                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;

                // 限制在图像边界内
                x1 = utils::clamp(x1, 0.0f, ipt_w);
                y1 = utils::clamp(y1, 0.0f, ipt_h);
                x2 = utils::clamp(x2, 0.0f, ipt_w);
                y2 = utils::clamp(y2, 0.0f, ipt_h);

                // 重新赋值到 box
                box.x = std::round(x1);
                box.y = std::round(y1);
                box.width = std::round(x2 - x1);
                box.height = std::round(y2 - y1);
            }
        }
        return true;
    }
}

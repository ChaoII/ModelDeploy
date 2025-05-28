//
// Created by aichao on 2025/2/20.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/detection/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsPostprocessor::UltralyticsPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
        multi_label_ = true;
        max_wh_ = 7680.0;
    }

    bool UltralyticsPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<DetectionResult>* results,
        const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) const {
        int batch = tensors[0].shape()[0];
        // transpose
        std::vector<int64_t> dim{0, 2, 1};
        Tensor tensor_transpose = tensors[0].transpose(dim).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            (*results)[bs].clear();
            if (multi_label_) {
                (*results)[bs].reserve(tensor_transpose.shape()[1] * (tensor_transpose.shape()[2] - 4));
            }
            else {
                (*results)[bs].reserve(tensor_transpose.shape()[1]);
            }
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const float* data =
                static_cast<const float*>(tensor_transpose.data()) +
                bs * tensor_transpose.shape()[1] * tensor_transpose.shape()[2];
            for (size_t i = 0; i < tensor_transpose.shape()[1]; ++i) {
                const int s = i * tensor_transpose.shape()[2];
                if (multi_label_) {
                    for (size_t j = 4; j < tensor_transpose.shape()[2]; ++j) {
                        float confidence = data[s + j];
                        // filter boxes by conf_threshold
                        if (confidence <= conf_threshold_) {
                            continue;
                        }
                        int32_t label_id = j - 4;
                        // convert from [x, y, w, h] to [x1, y1, x2, y2]
                        (*results)[bs].boxes.emplace_back(std::array<float, 4>{
                            data[s + 0] - data[s + 2] / 2.0f + label_id * max_wh_,
                            data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
                            data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
                            data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_
                        });
                        (*results)[bs].label_ids.push_back(label_id);
                        (*results)[bs].scores.push_back(confidence);
                    }
                }
                else {
                    const float* max_class_score = std::max_element(
                        data + s + 4, data + s + tensor_transpose.shape()[2]);
                    float confidence = *max_class_score;
                    // filter boxes by conf_threshold
                    if (confidence <= conf_threshold_) {
                        continue;
                    }
                    int32_t label_id = std::distance(data + s + 4, max_class_score);
                    // convert from [x, y, w, h] to [x1, y1, x2, y2]
                    (*results)[bs].boxes.emplace_back(std::array<float, 4>{
                        data[s + 0] - data[s + 2] / 2.0f + label_id * max_wh_,
                        data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
                        data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
                        data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_
                    });
                    (*results)[bs].label_ids.push_back(label_id);
                    (*results)[bs].scores.push_back(confidence);
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
            for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
                const int32_t label_id = (*results)[bs].label_ids[i];
                // clip box
                (*results)[bs].boxes[i][0] = (*results)[bs].boxes[i][0] - max_wh_ * label_id;
                (*results)[bs].boxes[i][1] = (*results)[bs].boxes[i][1] - max_wh_ * label_id;
                (*results)[bs].boxes[i][2] = (*results)[bs].boxes[i][2] - max_wh_ * label_id;
                (*results)[bs].boxes[i][3] = (*results)[bs].boxes[i][3] - max_wh_ * label_id;
                (*results)[bs].boxes[i][0] = std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
                (*results)[bs].boxes[i][1] = std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
                (*results)[bs].boxes[i][2] = std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
                (*results)[bs].boxes[i][3] = std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
                (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w);
                (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h);
                (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w);
                (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h);
            }
        }
        return true;
    }
}

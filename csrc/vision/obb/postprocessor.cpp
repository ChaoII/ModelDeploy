//
// Created by aichao on 2025/5/30.
//

#include <numeric>
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/obb/postprocessor.h"

#include <utils/utils.h>

namespace modeldeploy::vision::detection {
    UltralyticsObbPostprocessor::UltralyticsObbPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
    }

    bool UltralyticsObbPostprocessor::run_without_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<ObbResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,20,21504)->(1,21504,20) 20 = 4(xc,yc,w,h)+classes_num(15)+1(angle)
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 1}).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            // 官方模型为21504
            const auto dim1 = tensor_transpose.shape()[1];
            // 官方模型为4(xc, yc, w, h)+classes_num(15)+1(angle) = 20
            const auto dim2 = tensor_transpose.shape()[2];
            const float* data = static_cast<const float*>(tensor_transpose.data()) + bs * dim1 * dim2;
            std::vector<ObbResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                const auto attr_ptr = data + i * dim2;
                // 4(xc, yc, w, h)+classes_num(15)+1(angle)
                const float* max_class_score = std::max_element(attr_ptr + 4, attr_ptr + dim2 - 1);
                float confidence = *max_class_score;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(std::distance(attr_ptr + 4, max_class_score));
                // convert from [xc, yc, w, h, a]
                // 其中a为angle矩形框的旋转角度, 默认为弧度制(但是OpenCV的RotatedRect的旋转角度，默认为角度制)
                RotatedRect rotated_boxes = {
                    attr_ptr[0], attr_ptr[1],
                    attr_ptr[2], attr_ptr[3],
                    attr_ptr[dim2 - 1] * 180 / 3.141592653f
                };
                _results.push_back({rotated_boxes, label_id, confidence});
            }
            if (_results.empty()) {
                continue;
            }
            utils::obb_nms(&_results, nms_threshold_);
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            for (auto& result : _results) {
                auto& box = result.rotated_box;
                // clip box()
                //先减去 padding,再除以缩放因子scale;
                box.xc = (box.xc - pad_w) / scale;
                box.yc = (box.yc - pad_h) / scale;
                box.width = box.width / scale;
                box.height = box.height / scale;
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsObbPostprocessor::run_with_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<ObbResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,300,7)(xc, yc, w, h angle,score label_id)
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            // 官方模型为300
            const auto dim1 = tensors[0].shape()[1];
            // 官方模型为7 (xc, yc, w, h angle,score label_id)
            const auto dim2 = tensors[0].shape()[2];
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<ObbResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                const auto attr_ptr = data + i * dim2;
                float score = attr_ptr[5];
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(attr_ptr[6]);
                // convert from [xc, yc, w, h, a]
                // 其中a为angle矩形框的旋转角度, 默认为弧度制(但是OpenCV的RotatedRect的旋转角度，默认为角度制)
                RotatedRect rotated_boxes = {
                    attr_ptr[0], attr_ptr[1],
                    attr_ptr[2], attr_ptr[3],
                    attr_ptr[dim2 - 1] * 180 / 3.141592653f
                };
                _results.push_back({rotated_boxes, label_id, score});
            }
            if (_results.empty()) {
                continue;
            }
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            for (auto& result : _results) {
                auto& box = result.rotated_box;
                // clip box()
                //先减去 padding,再除以缩放因子scale;
                box.xc = (box.xc - pad_w) / scale;
                box.yc = (box.yc - pad_h) / scale;
                box.width = box.width / scale;
                box.height = box.height / scale;
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsObbPostprocessor::run(const std::vector<Tensor>& tensors,
                                          std::vector<std::vector<ObbResult>>* results,
                                          const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors[0].shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor." << std::endl;
            return false;
        }
        if (tensors[0].shape()[2] == 7) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}

//
// Created by aichao on 2025/2/20.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/detection/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsPostprocessor::UltralyticsPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
    }

    bool UltralyticsPostprocessor::run_without_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<DetectionResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,84,8400)->(1,8400,84) 84 = 4(xc,yc,w,h)+80(coco 80 classes)
        // 此处有一次数据拷贝
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 1}).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensor_transpose.shape()[1]; //8400
            const size_t dim2 = tensor_transpose.shape()[2]; //84
            const float* data = static_cast<const float*>(tensor_transpose.data()) + bs * dim1 * dim2;
            std::vector<DetectionResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                const float* max_class_score = std::max_element(attr_ptr + 4, attr_ptr + dim2);
                float confidence = *max_class_score;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) [[likely]] {
                    continue;
                }
                int32_t label_id = std::distance(attr_ptr + 4, max_class_score);
                // convert from [xc, yc, w, h] to [x, y, width, height]
                Rect2f box = {
                    attr_ptr[0] - attr_ptr[2] / 2.0f,
                    attr_ptr[1] - attr_ptr[3] / 2.0f,
                    attr_ptr[2],
                    attr_ptr[3]
                };
                _results.emplace_back(box, label_id, confidence);
            }
            if (_results.empty()) {
                continue;
            }
            utils::nms(&_results, nms_threshold_);
            // scale the boxes to the origin image shape

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                auto& box = result.box;
                // clip box()
                //1 先减去 padding;2除以缩放因子scale 3最后限制在原始图像范围内 [0, width], [0, height]。
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
                box.x = std::roundf(x1);
                box.y = std::roundf(y1);
                box.width = std::roundf(x2 - x1);
                box.height = std::roundf(y2 - y1);
            }
            (*results)[bs] = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPostprocessor::run_with_nms(const std::vector<Tensor>& tensors,
                                                std::vector<std::vector<DetectionResult>>* results,
                                                const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //300
            const size_t dim2 = tensors[0].shape()[2]; //6
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<DetectionResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                const float score = *(attr_ptr + 4);
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) [[likely]] {
                    continue;
                }
                int32_t label_id = *(attr_ptr + 5);
                // convert from [x1, y1, x2, y2] to [x, y, width, height]
                Rect2f box = {
                    attr_ptr[0],
                    attr_ptr[1],
                    attr_ptr[2] - attr_ptr[0],
                    attr_ptr[3] - attr_ptr[1]
                };
                _results.emplace_back(box, label_id, score);
            }
            if (_results.empty()) {
                continue;
            }
            // utils::nms(&_results, nms_threshold_);
            // // scale the boxes to the origin image shape

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                auto& box = result.box;
                // clip box()
                //1 先减去 padding;2除以缩放因子scale 3最后限制在原始图像范围内 [0, width], [0, height]。
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
                box.x = std::roundf(x1);
                box.y = std::roundf(y1);
                box.width = std::roundf(x2 - x1);
                box.height = std::roundf(y2 - y1);
            }
            (*results)[bs] = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPostprocessor::run(const std::vector<Tensor>& tensors,
                                       std::vector<std::vector<DetectionResult>>* results,
                                       const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors[0].shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor." << std::endl;
            return false;
        }
        if (tensors[0].shape()[2] == 6) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}

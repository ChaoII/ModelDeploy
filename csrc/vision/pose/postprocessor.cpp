//
// Created by aichao on 2025/06/2.
//

#include "core/md_log.h"
#include "vision/pose/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsPosePostprocessor::UltralyticsPosePostprocessor() {
        conf_threshold_ = 0.3;
        nms_threshold_ = 0.5;
        keypoints_num_ = 17;
    }

    bool UltralyticsPosePostprocessor::run_without_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<PoseResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        //  4(xc,yc,w,h)+1(conf)+17(keypoints)*3(x,y,conf)=56
        Tensor tensor_transpose = tensors[0].transpose({0, 2, 1}).to_tensor();
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            int64_t keypoints_num_by_output = (tensor_transpose.shape()[2] - 5) / 3;
            if (keypoints_num_by_output != keypoints_num_) {
                MD_LOG_ERROR << "Keypoints num set error, set" << keypoints_num_ << "but is: " <<
                    keypoints_num_by_output << "." << std::endl;
            }
            if (tensor_transpose.dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensor_transpose.shape()[1]; //?
            const size_t dim2 = tensor_transpose.shape()[2]; //56
            const float* data = static_cast<const float*>(tensor_transpose.data()) + bs * dim1 * dim2;

            std::vector<PoseResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                // 4(xc,yc,w,h)+1(conf)+17(keypoints)*3(x,y,conf)=56
                const float* attr_ptr = data + i * dim2;
                float confidence = attr_ptr[4];
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                int32_t label_id = 0;
                // convert from [xc, yc, w, h] to [x, y, width, height]
                Rect2f box = {
                    attr_ptr[0] - attr_ptr[2] / 2.0f,
                    attr_ptr[1] - attr_ptr[3] / 2.0f,
                    attr_ptr[2],
                    attr_ptr[3]
                };
                std::vector<Point3f> keypoints;
                keypoints.reserve(keypoints_num_);
                if (keypoints_num_ > 0) {
                    const float* keypoints_ptr = attr_ptr + 5;
                    for (size_t j = 0; j < keypoints_num_ * 3; j += 3) {
                        keypoints.emplace_back(
                            keypoints_ptr[j], keypoints_ptr[j + 1], keypoints_ptr[j + 2]);
                    }
                }
                _results.push_back({box, keypoints, label_id, confidence});
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
                for (auto& keypoint : result.keypoints) {
                    keypoint.x = (keypoint.x - pad_w) / scale;
                    keypoint.y = (keypoint.y - pad_h) / scale;
                }
                auto& box = result.box;
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
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPosePostprocessor::run_with_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<PoseResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        //  4(x1,y1,x2,y2)+1(conf)+1(class_id)+17(keypoints)*3(x,y,conf)=57
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            int64_t keypoints_num_by_output = (tensors[0].shape()[2] - 5) / 3;
            if (keypoints_num_by_output != keypoints_num_) {
                MD_LOG_ERROR << "Keypoints num set error, set" << keypoints_num_ << "but is: " <<
                    keypoints_num_by_output << "." << std::endl;
            }
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //300
            const size_t dim2 = tensors[0].shape()[2]; //6
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<PoseResult> _results;
            _results.reserve(tensors[0].shape()[1]);
            for (size_t i = 0; i < dim1; ++i) {
                // 4(x1,y1,x2,y2)+1(conf)+1(class_id)+17(keypoints)*3(x,y,conf)=57
                const float* attr_ptr = data + i * dim2;
                float score = attr_ptr[4];
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) {
                    continue;
                }
                int32_t label_id = attr_ptr[5];
                // convert from [x1, y1, x2, y2] to [x, y, width, height]
                Rect2f box = {
                    attr_ptr[0],
                    attr_ptr[1],
                    attr_ptr[2] - attr_ptr[0],
                    attr_ptr[3] - attr_ptr[1]
                };
                std::vector<Point3f> keypoints;
                keypoints.reserve(keypoints_num_);
                if (keypoints_num_ > 0) {
                    const float* keypoints_ptr = attr_ptr + 6;
                    for (size_t j = 0; j < keypoints_num_ * 3; j += 3) {
                        keypoints.emplace_back(
                            keypoints_ptr[j], keypoints_ptr[j + 1], keypoints_ptr[j + 2]);
                    }
                }
                _results.push_back({box, keypoints, label_id, score});
            }
            if (_results.empty()) {
                continue;
            }
            // scale the boxes to the origin image shape
            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                for (auto& keypoint : result.keypoints) {
                    keypoint.x = (keypoint.x - pad_w) / scale;
                    keypoint.y = (keypoint.y - pad_h) / scale;
                }
                auto& box = result.box;
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
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPosePostprocessor::run(const std::vector<Tensor>& tensors,
                                           std::vector<std::vector<PoseResult>>* results,
                                           const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors[0].shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor." << std::endl;
            return false;
        }
        if (tensors[0].shape()[2] == 57) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}

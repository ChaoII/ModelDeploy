//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_det/postprocessor.h"

namespace modeldeploy::vision::face {
    ScrfdPostprocessor::ScrfdPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
    }

    void ScrfdPostprocessor::generate_points(const int width, const int height) {
        if (center_points_is_update_ && !is_dynamic_input_) {
            return;
        }
        // 8, 16, 32
        for (auto local_stride : downsample_strides_) {
            const unsigned int num_grid_w = width / local_stride;
            const unsigned int num_grid_h = height / local_stride;
            // y
            for (unsigned int i = 0; i < num_grid_h; ++i) {
                // x
                for (unsigned int j = 0; j < num_grid_w; ++j) {
                    // num_anchors, col major
                    for (unsigned int k = 0; k < num_anchors_; ++k) {
                        SCRFDPoint point;
                        point.cx = static_cast<float>(j);
                        point.cy = static_cast<float>(i);
                        center_points_[local_stride].push_back(point);
                    }
                }
            }
        }
        center_points_is_update_ = true;
    }


    bool ScrfdPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<std::vector<DetectionLandmarkResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) {
        const size_t fmc = downsample_strides_.size();
        // scrfd has 6,9,10,15 output tensors
        if (!(tensors.size() == 9 || tensors.size() == 6 ||
            tensors.size() == 10 || tensors.size() == 15)) {
            MD_LOG_ERROR << "The default number of output tensor must be 6, 9, 10, or 15 "
                "according to scrfd." << std::endl;
            return false;
        }
        if (!(fmc == 3 || fmc == 5)) { MD_LOG_ERROR << "The fmc must be 3 or 5" << std::endl; }
        if (tensors.at(0).shape()[0] != 1) {
            MD_LOG_ERROR << "Only support batch =1 now." << std::endl;
            return false;
        }
        for (int i = 0; i < fmc; ++i) {
            if (tensors.at(i).dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
        }
        size_t total_num_boxes = 0;
        // compute the reserve space.
        for (int f = 0; f < fmc; ++f) {
            total_num_boxes += tensors.at(f).shape()[1];
        }
        const size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float out_h = letter_box_records[bs].out_h;
            const float out_w = letter_box_records[bs].out_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            generate_points(out_w, out_h);
            std::vector<DetectionLandmarkResult> _results;
            _results.reserve(static_cast<int>(total_num_boxes));
            unsigned int count = 0;
            // loop each stride
            for (int f = 0; f < fmc; ++f) {
                const auto* score_ptr = static_cast<const float*>(tensors.at(f).data());
                const auto* bbox_ptr = static_cast<const float*>(tensors.at(f + fmc).data());
                const unsigned int num_points = tensors.at(f).shape()[1];
                int current_stride = downsample_strides_[f];
                auto& stride_points = center_points_[current_stride];
                // loop each anchor
                for (unsigned int i = 0; i < num_points; ++i) {
                    const float cls_conf = score_ptr[i];
                    if (cls_conf < conf_threshold_) continue; // filter
                    const auto& point = stride_points.at(i);
                    const float cx = point.cx; // cx
                    const float cy = point.cy; // cy
                    // bbox
                    const float* offsets = bbox_ptr + i * 4;
                    const float l = offsets[0]; // left
                    const float t = offsets[1]; // top
                    const float r = offsets[2]; // right
                    const float b = offsets[3]; // bottom

                    const float x1 = ((cx - l) * static_cast<float>(current_stride) - pad_w) / scale; // cx - l x1
                    const float y1 = ((cy - t) * static_cast<float>(current_stride) - pad_h) / scale; // cy - t y1
                    const float x2 = ((cx + r) * static_cast<float>(current_stride) - pad_w) / scale; // cx + r x2
                    const float y2 = ((cy + b) * static_cast<float>(current_stride) - pad_h) / scale; // cy + b y2
                    std::vector<Point2f> landmarks;
                    landmarks.reserve(landmarks_per_face_);
                    if (use_kps_) {
                        const auto* landmarks_ptr =
                            static_cast<const float*>(tensors.at(f + 2 * fmc).data());
                        // landmarks
                        const float* kps_offsets = landmarks_ptr + i * (landmarks_per_face_ * 2);
                        for (unsigned int j = 0; j < landmarks_per_face_ * 2; j += 2) {
                            const float kps_l = kps_offsets[j];
                            const float kps_t = kps_offsets[j + 1];
                            const float kps_x = ((cx + kps_l) * static_cast<float>(current_stride) - pad_w) / scale;
                            // cx + l x
                            const float kps_y = ((cy + kps_t) * static_cast<float>(current_stride) - pad_h) / scale;
                            // cy + t y
                            landmarks.emplace_back(kps_x, kps_y);
                        }
                    }
                    _results.push_back({Rect2f{x1, y1, x2 - x1, y2 - y1}, landmarks, 0, cls_conf});
                    count += 1; // limit boxes for nms.
                    if (count > max_nms_) {
                        break;
                    }
                }
            }

            if (_results.empty()) {
                return true;
            }
            utils::nms(&_results, nms_threshold_);
            // scale and clip box
            for (auto& _result : _results) {
                auto& box = _result.box;
                auto& landmarks = _result.landmarks;
                // 左上角不能越界
                box.x = std::max(box.x, 0.0f);
                box.y = std::max(box.y, 0.0f);

                // 右下角也不能越界
                const float x2 = std::min(box.x + box.width, ipt_w - 1.0f);
                const float y2 = std::min(box.y + box.height, ipt_h - 1.0f);

                // 防止裁剪后右下角比左上角还小
                box.width = std::max(x2 - box.x, 0.0f);
                box.height = std::max(y2 - box.y, 0.0f);
                if (use_kps_) {
                    for (auto& landmark : landmarks) {
                        landmark.x = std::max(landmark.x, 0.0f);
                        landmark.y = std::max(landmark.y, 0.0f);
                        landmark.x = std::min(landmark.x, ipt_w - 1.0f);
                        landmark.y = std::min(landmark.y, ipt_h - 1.0f);
                    }
                }
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }
}

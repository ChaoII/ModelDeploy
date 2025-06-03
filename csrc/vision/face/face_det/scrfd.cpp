//
// Created by aichao on 2025/3/26.
//
#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/convert.h"
#include "csrc/vision/face/face_det/scrfd.h"


namespace modeldeploy::vision::face {
    void SCRFD::letter_box(cv::Mat* mat, const std::vector<int>& size,
                           const std::vector<float>& color, const bool _auto,
                           const bool scale_fill, const bool scale_up, const int stride) {
        auto scale = std::min(size[1] * 1.0 / mat->rows, size[0] * 1.0 / mat->cols);
        if (!scale_up) {
            scale = std::min(scale, 1.0);
        }

        int resize_h = static_cast<int>(round(mat->rows * scale));
        int resize_w = static_cast<int>(round(mat->cols * scale));

        int pad_w = size[0] - resize_w;
        int pad_h = size[1] - resize_h;
        if (_auto) {
            pad_h = pad_h % stride;
            pad_w = pad_w % stride;
        }
        else if (scale_fill) {
            pad_h = 0;
            pad_w = 0;
            resize_h = size[1];
            resize_w = size[0];
        }
        if (resize_h != mat->rows || resize_w != mat->cols) {
            Resize::apply(mat, resize_w, resize_h);
        }
        if (pad_h > 0 || pad_w > 0) {
            const float half_h = static_cast<float>(pad_h) * 1.0f / 2;
            const int top = static_cast<int>(round(half_h - 0.1));
            const int bottom = static_cast<int>(round(half_h + 0.1));
            const float half_w = static_cast<float>(pad_w) * 1.0f / 2;
            const int left = static_cast<int>(round(half_w - 0.1));
            const int right = static_cast<int>(round(half_w + 0.1));
            Pad::apply(mat, top, bottom, left, right, color);
        }
    }

    SCRFD::SCRFD(const std::string& model_file,
                 const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool SCRFD::Initialize() {
        // num_outputs = use_kps ? 9 : 6;
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        // Check if the input shape is dynamic after Runtime already initialized,
        // Note that, We need to force is_mini_pad 'false' to keep static
        // shape after padding (LetterBox) when the is_dynamic_shape is 'false'.
        is_dynamic_input_ = false;
        const auto shape = get_input_info(0).shape;
        for (int i = 0; i < shape.size(); ++i) {
            // if height or width is dynamic
            if (i >= 2 && shape[i] <= 0) {
                is_dynamic_input_ = true;
                break;
            }
        }
        if (!is_dynamic_input_) {
            is_mini_pad = false;
        }

        return true;
    }

    bool SCRFD::preprocess(cv::Mat* mat, Tensor* output,
                           std::map<std::string, std::array<float, 2>>* im_info) {
        const float ratio = std::min(
            static_cast<float>(size[1]) * 1.0f / static_cast<float>(mat->rows),
            static_cast<float>(size[0]) * 1.0f / static_cast<float>(mat->cols));

        if (std::fabs(ratio - 1.0f) > 1e-06) {
            int interp = cv::INTER_LINEAR;
            if (ratio > 1.0) {
                interp = cv::INTER_LINEAR;
            }
            const int resize_h = static_cast<int>(static_cast<float>(mat->rows) * ratio);
            const int resize_w = static_cast<int>(static_cast<float>(mat->cols) * ratio);
            Resize::apply(mat, resize_w, resize_h, -1, -1, interp);
        }
        // scrfd's preprocess steps
        // 1. letterbox
        // 2. BGR->RGB
        // 3. HWC->CHW
        SCRFD::letter_box(mat, size, padding_value, is_mini_pad, is_no_pad, is_scale_up, stride);

        BGR2RGB::apply(mat);

        // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
        //                std::vector<float>(mat->Channels(), 1.0));
        // Compute `result = mat * alpha + beta` directly by channel
        // Original Repo/tools/scrfd.py: cv2.dnn.blobFromImage(img, 1.0/128,
        // input_size, (127.5, 127.5, 127.5), swapRB=True)
        const std::vector alpha = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
        const std::vector beta = {-127.5f / 128.f, -127.5f / 128.f, -127.5f / 128.f};
        Convert::apply(mat, alpha, beta);
        HWC2CHW::apply(mat);
        Cast::apply(mat, "float");

        // Record output shape of preprocessed image
        (*im_info)["output_shape"] = {
            static_cast<float>(mat->rows),
            static_cast<float>(mat->cols)
        };
        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    void SCRFD::generate_points() {
        if (center_points_is_update_ && !is_dynamic_input_) {
            return;
        }
        // 8, 16, 32
        for (auto local_stride : downsample_strides) {
            const unsigned int num_grid_w = size[0] / local_stride;
            const unsigned int num_grid_h = size[1] / local_stride;
            // y
            for (unsigned int i = 0; i < num_grid_h; ++i) {
                // x
                for (unsigned int j = 0; j < num_grid_w; ++j) {
                    // num_anchors, col major
                    for (unsigned int k = 0; k < num_anchors; ++k) {
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

    bool SCRFD::postprocess(
        std::vector<Tensor>& infer_result, std::vector<DetectionLandmarkResult>* result,
        const std::map<std::string, std::array<float, 2>>& im_info,
        const float conf_threshold, const float nms_iou_threshold) {
        // number of downsample_strides
        const size_t fmc = downsample_strides.size();
        // scrfd has 6,9,10,15 output tensors
        if (!(infer_result.size() == 9 || infer_result.size() == 6 ||
            infer_result.size() == 10 || infer_result.size() == 15)) {
            MD_LOG_ERROR << "The default number of output tensor must be 6, 9, 10, or 15 "
                "according to scrfd." << std::endl;
            return false;
        }
        if (!(fmc == 3 || fmc == 5)) { MD_LOG_ERROR << "The fmc must be 3 or 5" << std::endl; }
        if (infer_result.at(0).shape()[0] != 1) {
            MD_LOG_ERROR << "Only support batch =1 now." << std::endl;
            return false;
        }
        for (int i = 0; i < fmc; ++i) {
            if (infer_result.at(i).dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
        }
        size_t total_num_boxes = 0;
        // compute the reserve space.
        for (int f = 0; f < fmc; ++f) {
            total_num_boxes += infer_result.at(f).shape()[1];
        }
        generate_points();
        result->clear();
        // scale the boxes to the origin image shape
        const auto iter_out = im_info.find("output_shape");
        const auto iter_ipt = im_info.find("input_shape");

        if (!(iter_out != im_info.end() && iter_ipt != im_info.end())) {
            MD_LOG_ERROR << "Cannot find input_shape or output_shape from im_info." << std::endl;
        }
        const float out_h = iter_out->second[0];
        const float out_w = iter_out->second[1];
        const float ipt_h = iter_ipt->second[0];
        const float ipt_w = iter_ipt->second[1];

        float scale = std::min(out_h / ipt_h, out_w / ipt_w);
        if (!is_scale_up) {
            scale = std::min(scale, 1.0f);
        }
        float pad_h = (out_h - ipt_h * scale) / 2.0f;
        float pad_w = (out_w - ipt_w * scale) / 2.0f;
        if (is_mini_pad) {
            pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
            pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
        }
        result->reserve(static_cast<int>(total_num_boxes));
        unsigned int count = 0;
        // loop each stride
        for (int f = 0; f < fmc; ++f) {
            const auto* score_ptr = static_cast<float*>(infer_result.at(f).data());
            const auto* bbox_ptr = static_cast<float*>(infer_result.at(f + fmc).data());
            const unsigned int num_points = infer_result.at(f).shape()[1];
            int current_stride = downsample_strides[f];
            auto& stride_points = center_points_[current_stride];
            // loop each anchor
            for (unsigned int i = 0; i < num_points; ++i) {
                const float cls_conf = score_ptr[i];
                if (cls_conf < conf_threshold) continue; // filter
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
                std::vector<cv::Point2f> landmarks;
                landmarks.reserve(landmarks_per_face);
                if (use_kps) {
                    const auto* landmarks_ptr =
                        static_cast<float*>(infer_result.at(f + 2 * fmc).data());
                    // landmarks
                    const float* kps_offsets = landmarks_ptr + i * (landmarks_per_face * 2);
                    for (unsigned int j = 0; j < landmarks_per_face * 2; j += 2) {
                        const float kps_l = kps_offsets[j];
                        const float kps_t = kps_offsets[j + 1];
                        const float kps_x = ((cx + kps_l) * static_cast<float>(current_stride) - pad_w) / scale;
                        // cx + l x
                        const float kps_y = ((cy + kps_t) * static_cast<float>(current_stride) - pad_h) / scale;
                        // cy + t y
                        landmarks.emplace_back(kps_x, kps_y);
                    }
                }
                result->emplace_back(cv::Rect2f{x1, y1, x2 - x1, y2 - y1}, landmarks, cls_conf, 0);
                count += 1; // limit boxes for nms.
                if (count > max_nms) {
                    break;
                }
            }
        }

        // fetch original image shape
        if (iter_ipt == im_info.end()) {
            MD_LOG_ERROR << "Cannot find input_shape from im_info." << std::endl;
        }
        if (result->empty()) {
            return true;
        }
        utils::nms(result, nms_iou_threshold);
        // scale and clip box
        for (auto& _result : *result) {
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
            if (use_kps) {
                for (auto& landmark : landmarks) {
                    landmark.x = std::max(landmark.x, 0.0f);
                    landmark.y = std::max(landmark.y, 0.0f);
                    landmark.x = std::min(landmark.x, ipt_w - 1.0f);
                    landmark.y = std::min(landmark.y, ipt_h - 1.0f);
                }
            }
        }
        // scale and clip landmarks
        return true;
    }

    bool SCRFD::predict(cv::Mat& im, std::vector<DetectionLandmarkResult>* result,
                        const float conf_threshold, const float nms_iou_threshold) {
        std::vector<Tensor> input_tensors(1);
        std::map<std::string, std::array<float, 2>> im_info;
        // Record the shape of image and the shape of preprocessed image
        im_info["input_shape"] = {
            static_cast<float>(im.rows),
            static_cast<float>(im.cols)
        };
        im_info["output_shape"] = {
            static_cast<float>(im.rows),
            static_cast<float>(im.cols)
        };
        if (!preprocess(&im, &input_tensors[0], &im_info)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }

        input_tensors[0].set_name(get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }

        if (!postprocess(output_tensors, result, im_info, conf_threshold,
                         nms_iou_threshold)) {
            MD_LOG_ERROR << "Failed to post process." << std::endl;
            return false;
        }
        return true;
    }
}

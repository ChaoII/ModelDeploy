#include "csrc/core/md_log.h"
#include "csrc/vision/lpr/lpr_det/lpr_det.h"
#include "csrc/vision/utils.h"
#include "csrc/utils/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/convert.h"


namespace modeldeploy::vision::lpr {
    void LetterBox(cv::Mat* mat, const std::vector<int>& size, const std::vector<float>& color,
                   const bool _auto, const bool scale_fill = false, const bool scale_up = true, const int stride = 32) {
        float scale = std::min(size[1] * 1.0f / mat->rows, size[0] * 1.0f / mat->cols);
        if (!scale_up) {
            scale = std::min(scale, 1.0f);
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

    LprDetection::LprDetection(const std::string& model_file,
                               const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool LprDetection::initialize() {
        // parameters for preprocess
        size = {640, 640};
        padding_value = {114.0, 114.0, 114.0};
        is_mini_pad = false;
        is_no_pad = false;
        is_scale_up = true;
        stride = 32;
        landmarks_per_card = 4;

        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        // Check if the input shape is dynamic after Runtime already initialized,
        // Note that, We need to force is_mini_pad 'false' to keep static
        // shape after padding (LetterBox) when the is_dynamic_input_ is 'false'.
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

    bool LprDetection::preprocess(
        cv::Mat& image, Tensor* output,
        std::map<std::string, std::array<float, 2>>* im_info) {
        // process after image load

        const float ratio = std::min(static_cast<float>(size[1]) * 1.0f / static_cast<float>(image.rows),
                                     static_cast<float>(size[0]) * 1.0f / static_cast<float>(image.cols));

        if (std::fabs(ratio - 1.0f) > 1e-06) {
            int interp = cv::INTER_LINEAR;
            if (ratio > 1.0) {
                interp = cv::INTER_LINEAR;
            }
            const int resize_h = static_cast<int>(round(static_cast<float>(image.rows) * ratio));
            const int resize_w = static_cast<int>(round(static_cast<float>(image.cols) * ratio));
            Resize::apply(&image, resize_w, resize_h, -1, -1, interp);
        }

        // yolov5face's preprocess steps
        // 1. letterbox
        // 2. BGR->RGB
        // 3. HWC->CHW
        LetterBox(&image, size, padding_value, is_mini_pad, is_no_pad, is_scale_up,
                  stride);
        BGR2RGB::apply(&image);
        // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
        //                std::vector<float>(mat->Channels(), 1.0));
        // Compute `result = mat * alpha + beta` directly by channel
        const std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector beta = {0.0f, 0.0f, 0.0f};
        Convert::apply(&image, alpha, beta);

        // Record output shape of preprocessed image
        (*im_info)["output_shape"] = {
            static_cast<float>(image.rows),
            static_cast<float>(image.cols)
        };

        HWC2CHW::apply(&image);
        Cast::apply(&image, "float");

        if (!utils::mat_to_tensor(image, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool LprDetection::postprocess(
        const Tensor& infer_result, std::vector<DetectionLandmarkResult>* result,
        const std::map<std::string, std::array<float, 2>>& im_info,
        const float conf_threshold, const float nms_iou_threshold) const {
        // infer_result: (1,n,15) 15=4+1+8+1+1
        if (infer_result.shape()[0] != 1) {
            MD_LOG_ERROR << "Only support batch =1 now." << std::endl;
        }
        if (infer_result.dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        result->clear();
        result->reserve(static_cast<int>(infer_result.shape()[1]));
        auto* data = static_cast<const float*>(infer_result.data());
        // x,y,w,h,obj_conf,x1,y1,x2,y2,x3,y3,x4,y4,cls_conf0(单层车牌),cls_conf1(双层车牌)
        for (size_t i = 0; i < infer_result.shape()[1]; ++i) {
            const float* attr_ptr = data + i * infer_result.shape()[2];
            const float obj_conf = attr_ptr[4];
            // const float cls_conf = reg_cls_ptr[13];
            // 0: 单层车牌
            // 1: 双层车牌
            std::vector<float> cls_confs = {attr_ptr + 13, attr_ptr + infer_result.shape()[2]};
            const int class_id = argmax(cls_confs);
            const auto cls_conf = *std::ranges::max_element(cls_confs);
            float confidence = obj_conf * cls_conf;
            // filter boxes by conf_threshold
            if (confidence <= conf_threshold) {
                continue;
            }

            const float x = attr_ptr[0];
            const float y = attr_ptr[1];
            const float w = attr_ptr[2];
            const float h = attr_ptr[3];


            std::vector<cv::Point2f> landmarks;
            landmarks.reserve(landmarks_per_card);
            // decode landmarks (default 5 landmarks)
            if (landmarks_per_card > 0) {
                const float* landmarks_ptr = attr_ptr + 5;
                for (size_t j = 0; j < landmarks_per_card * 2; j += 2) {
                    landmarks.emplace_back(
                        landmarks_ptr[j], landmarks_ptr[j + 1]);
                }
            }
            result->emplace_back(
                cv::Rect2f{x - w / 2.f, y - h / 2.f, w, h},
                landmarks, class_id, confidence);
        }
        if (result->empty()) {
            return true;
        }
        utils::nms(result, nms_iou_threshold);
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
        const float scale = std::min(out_h / ipt_h, out_w / ipt_w);
        const float pad_h = (out_h - ipt_h * scale) / 2;
        const float pad_w = (out_w - ipt_w * scale) / 2;
        // scale and clip box
        for (auto& _result : *result) {
            auto& box = _result.box;
            auto& landmarks = _result.landmarks;
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

            // scale and clip landmarks
            for (auto& landmark : landmarks) {
                landmark.x = std::max((landmark.x - pad_w) / scale, 0.0f);
                landmark.y = std::max((landmark.y - pad_h) / scale, 0.0f);

                landmark.x = std::min(landmark.x, ipt_w - 1.0f);
                landmark.y = std::min(landmark.y, ipt_h - 1.0f);
            }
        }

        return true;
    }

    bool LprDetection::predict(const cv::Mat& image, std::vector<DetectionLandmarkResult>* result,
                               const float conf_threshold, const float nms_iou_threshold) {
        std::vector<Tensor> input_tensors(1);
        std::map<std::string, std::array<float, 2>> im_info;
        // Record the shape of image and the shape of preprocessed image
        im_info["input_shape"] = {
            static_cast<float>(image.rows),
            static_cast<float>(image.cols)
        };
        im_info["output_shape"] = {
            static_cast<float>(image.rows),
            static_cast<float>(image.cols)
        };

        // 预处理会修改image的shape等等相关数据，为了保证原始图像不被修改，需要拷贝一份图像数据
        cv::Mat image_bak = image;
        if (!preprocess(image_bak, &input_tensors[0], &im_info)) {
            MD_LOG_ERROR << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].set_name(get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (!postprocess(output_tensors[0], result, im_info, conf_threshold,
                         nms_iou_threshold)) {
            MD_LOG_ERROR << "Failed to post process." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::facedet

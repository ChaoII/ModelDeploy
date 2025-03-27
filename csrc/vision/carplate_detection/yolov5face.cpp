#include "csrc/core/md_log.h"
#include "csrc/vision/carplate_detection/yolov5face.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/convert.h"


namespace modeldeploy::vision::facedet {
    void LetterBox(cv::Mat* mat, std::vector<int> size, std::vector<float> color,
                   bool _auto, bool scale_fill = false, bool scale_up = true,
                   int stride = 32) {
        float scale =
            std::min(size[1] * 1.0 / mat->rows, size[0] * 1.0 / mat->cols);
        if (!scale_up) {
            scale = std::min(scale, 1.0f);
        }

        int resize_h = int(round(mat->rows * scale));
        int resize_w = int(round(mat->cols * scale));

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
            Resize::Run(mat, resize_w, resize_h);
        }
        if (pad_h > 0 || pad_w > 0) {
            float half_h = pad_h * 1.0 / 2;
            int top = int(round(half_h - 0.1));
            int bottom = int(round(half_h + 0.1));
            float half_w = pad_w * 1.0 / 2;
            int left = int(round(half_w - 0.1));
            int right = int(round(half_w + 0.1));
            Pad::Run(mat, top, bottom, left, right, color);
        }
    }

    YOLOv5Face::YOLOv5Face(const std::string& model_file,
                           const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool YOLOv5Face::Initialize() {
        // parameters for preprocess
        size = {640, 640};
        padding_value = {114.0, 114.0, 114.0};
        is_mini_pad = false;
        is_no_pad = false;
        is_scale_up = true;
        stride = 32;
        landmarks_per_face = 4;

        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        // Check if the input shape is dynamic after Runtime already initialized,
        // Note that, We need to force is_mini_pad 'false' to keep static
        // shape after padding (LetterBox) when the is_dynamic_input_ is 'false'.
        is_dynamic_input_ = false;
        auto shape = get_input_info(0).shape;
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

    bool YOLOv5Face::Preprocess(
        cv::Mat* mat, MDTensor* output,
        std::map<std::string, std::array<float, 2>>* im_info) {
        // process after image load
        float ratio = std::min(size[1] * 1.0f / static_cast<float>(mat->rows),
                               size[0] * 1.0f / static_cast<float>(mat->cols));

        if (std::fabs(ratio - 1.0f) > 1e-06) {
            int interp = cv::INTER_LINEAR;
            if (ratio > 1.0) {
                interp = cv::INTER_LINEAR;
            }
            int resize_h = int(round(static_cast<float>(mat->rows) * ratio));
            int resize_w = int(round(static_cast<float>(mat->cols) * ratio));
            Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
        }

        // yolov5face's preprocess steps
        // 1. letterbox
        // 2. BGR->RGB
        // 3. HWC->CHW
        LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad, is_scale_up,
                  stride);
        BGR2RGB::Run(mat);
        // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
        //                std::vector<float>(mat->Channels(), 1.0));
        // Compute `result = mat * alpha + beta` directly by channel
        std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        std::vector<float> beta = {0.0f, 0.0f, 0.0f};
        Convert::Run(mat, alpha, beta);

        // Record output shape of preprocessed image
        (*im_info)["output_shape"] = {
            static_cast<float>(mat->rows),
            static_cast<float>(mat->cols)
        };

        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");

        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR("Failed to binding mat to tensor.");
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool YOLOv5Face::Postprocess(
        MDTensor& infer_result, FaceDetectionResult* result,
        const std::map<std::string, std::array<float, 2>>& im_info,
        float conf_threshold, float nms_iou_threshold) {
        // infer_result: (1,n,14) 15=4+1+8+1
        if (infer_result.shape[0] != 1) { std::cerr << "Only support batch =1 now." << std::endl; }
        if (infer_result.dtype != MDDataType::Type::FP32) {
            std::cerr << "Only support post process with float32 data." << std::endl;
            return false;
        }
        result->Clear();
        // must be setup landmarks_per_face before reserve
        result->landmarks_per_face = landmarks_per_face;
        result->Reserve(infer_result.shape[1]);

        float* data = static_cast<float*>(infer_result.data());
        // x,y,w,h,obj_conf,x1,y1,x2,y2,x3,y3,x4,y4,cls_conf
        for (size_t i = 0; i < infer_result.shape[1]; ++i) {
            float* reg_cls_ptr = data + i * infer_result.shape[2];
            float obj_conf = reg_cls_ptr[4];
            float cls_conf = reg_cls_ptr[13];
            float confidence = obj_conf * cls_conf;
            // filter boxes by conf_threshold
            if (confidence <= conf_threshold) {
                continue;
            }
            float x = reg_cls_ptr[0];
            float y = reg_cls_ptr[1];
            float w = reg_cls_ptr[2];
            float h = reg_cls_ptr[3];

            // convert from [x, y, w, h] to [x1, y1, x2, y2]
            result->boxes.emplace_back(std::array<float, 4>{
                (x - w / 2.f), (y - h / 2.f), (x + w / 2.f), (y + h / 2.f)
            });
            result->scores.push_back(confidence);
            // decode landmarks (default 5 landmarks)
            if (landmarks_per_face > 0) {
                float* landmarks_ptr = reg_cls_ptr + 5;
                for (size_t j = 0; j < landmarks_per_face * 2; j += 2) {
                    result->landmarks.emplace_back(
                        std::array<float, 2>{landmarks_ptr[j], landmarks_ptr[j + 1]});
                }
            }
        }

        if (result->boxes.size() == 0) {
            return true;
        }

        utils::nms(result, nms_iou_threshold);

        // scale the boxes to the origin image shape
        auto iter_out = im_info.find("output_shape");
        auto iter_ipt = im_info.find("input_shape");
        if (!(iter_out != im_info.end() && iter_ipt != im_info.end())) {
            std::cerr << "Cannot find input_shape or output_shape from im_info." << std::endl;
        }
        float out_h = iter_out->second[0];
        float out_w = iter_out->second[1];
        float ipt_h = iter_ipt->second[0];
        float ipt_w = iter_ipt->second[1];
        float scale = std::min(out_h / ipt_h, out_w / ipt_w);
        if (!is_scale_up) {
            scale = std::min(scale, 1.0f);
        }
        float pad_h = (out_h - ipt_h * scale) / 2.f;
        float pad_w = (out_w - ipt_w * scale) / 2.f;
        if (is_mini_pad) {
            pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
            pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
        }
        // scale and clip box
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / scale, 0.0f);
            result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / scale, 0.0f);
            result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / scale, 0.0f);
            result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / scale, 0.0f);
            result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
            result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
            result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
            result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
        }
        // scale and clip landmarks
        for (size_t i = 0; i < result->landmarks.size(); ++i) {
            result->landmarks[i][0] =
                std::max((result->landmarks[i][0] - pad_w) / scale, 0.0f);
            result->landmarks[i][1] =
                std::max((result->landmarks[i][1] - pad_h) / scale, 0.0f);
            result->landmarks[i][0] = std::min(result->landmarks[i][0], ipt_w - 1.0f);
            result->landmarks[i][1] = std::min(result->landmarks[i][1], ipt_h - 1.0f);
        }
        return true;
    }

    bool YOLOv5Face::Predict(cv::Mat* image, FaceDetectionResult* result,
                             float conf_threshold, float nms_iou_threshold) {
        std::vector<MDTensor> input_tensors(1);

        std::map<std::string, std::array<float, 2>> im_info;

        // Record the shape of image and the shape of preprocessed image
        im_info["input_shape"] = {
            static_cast<float>(image->rows),
            static_cast<float>(image->cols)
        };
        im_info["output_shape"] = {
            static_cast<float>(image->rows),
            static_cast<float>(image->cols)
        };

        if (!Preprocess(image, &input_tensors[0], &im_info)) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].name = get_input_info(0).name;
        std::vector<MDTensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            std::cerr << "Failed to inference." << std::endl;
            return false;
        }

        if (!Postprocess(output_tensors[0], result, im_info, conf_threshold,
                         nms_iou_threshold)) {
            std::cerr << "Failed to post process." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::facedet

// namespace fastdeploy

#include "csrc/core/md_log.h"
#include "csrc/vision/carplate_recognizer/carplate_recognizer.h"
#include "csrc/vision/utils.h"
#include "csrc/utils/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/pad.h"
#include "csrc/vision/common/processors/convert.h"


namespace modeldeploy::vision::facedet {

    CarplateRecognizer::CarplateRecognizer(const std::string& model_file,
                           const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool CarplateRecognizer::Initialize() {
        // parameters for preprocess
        size = {168, 48};

        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool CarplateRecognizer::Preprocess(
        cv::Mat* mat, MDTensor* output,
        std::map<std::string, std::array<float, 2>>* im_info) {

        // yolov5face's preprocess steps
        // 1. resize
        // 2. BGR->RGB
        // 3. HWC->CHW

        Resize::Run(mat, size[0],size[1]);
        BGR2RGB::Run(mat);
        // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
        //                std::vector<float>(mat->Channels(), 1.0));
        // Compute `result = mat * alpha + beta` directly by channel
        std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        std::vector<float> beta = { -0.588f,  -0.588f,  -0.588f};
        Convert::Run(mat, alpha, beta);

        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");

        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR("Failed to binding mat to tensor.");
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool CarplateRecognizer::Postprocess(
        std::vector<MDTensor>& infer_result, FaceDetectionResult* result,
        const std::map<std::string, std::array<float, 2>>& im_info,
        float conf_threshold, float nms_iou_threshold) {
        std::vector<float>plate_color_vec(5);
        memcpy(plate_color_vec.data(),infer_result[1].data(),5*sizeof(float));
        int max_Index=argmax(plate_color_vec);
        std::string plate_color=plate_color_list[max_Index];
        //车牌
        std::vector<int>plate_index;
        std::vector<float>plate_tensor(78);
        float* prob1_temp=static_cast<float*> (infer_result[0].data());
        for (size_t j = 0; j < 21; j++)
        {
            memcpy(plate_tensor.data(),prob1_temp,78*sizeof(float));
            int max_Index=argmax(plate_tensor);
            plate_index.push_back(max_Index);
            prob1_temp=prob1_temp+78;
        }
        int pre=0;
        std::string plate_str;
        for (size_t j = 0; j < plate_index.size(); j++)
        {
            if(plate_index[j]!=0&&plate_index[j]!=pre)
            {
                plate_str+=plate_chr[plate_index[j]];
            }
            pre=plate_index[j];
        }

        std::cout<<"plate_str: "<<plate_str<<std::endl;
        std::cout<<"color: "<<plate_color<<std::endl;

        return true;
    }

    bool CarplateRecognizer::Predict(cv::Mat* image, FaceDetectionResult* result,
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

        if (!Postprocess(output_tensors, result, im_info, conf_threshold,
                         nms_iou_threshold)) {
            std::cerr << "Failed to post process." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::facedet

// namespace fastdeploy

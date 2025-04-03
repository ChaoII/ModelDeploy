#include "csrc/core/md_log.h"
#include "csrc/vision/lpr/lpr_rec/lpr_rec.h"
#include "csrc/vision/utils.h"
#include "csrc/utils/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/convert.h"


namespace modeldeploy::vision::lpr {
    LprRecognizer::LprRecognizer(const std::string& model_file,
                                 const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool LprRecognizer::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool LprRecognizer::preprocess(
        cv::Mat& mat, MDTensor* output) {
        // car_plate_recognizer's preprocess steps
        // 1. resize
        // 2. BGR->RGB
        // 3. HWC->CHW

        Resize::Run(&mat, size[0], size[1]);
        BGR2RGB::Run(&mat);
        // Compute `result = mat * alpha + beta` directly by channel
        std::vector alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        std::vector beta = {-0.588f, -0.588f, -0.588f};
        Convert::Run(&mat, alpha, beta);
        HWC2CHW::Run(&mat);
        Cast::Run(&mat, "float");
        if (!utils::mat_to_tensor(mat, output)) {
            MD_LOG_ERROR << "Failed to binding mat to tensor." << std::endl;
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool LprRecognizer::postprocess(
        std::vector<MDTensor>& infer_result, LprResult* result) {
        auto* plate_color_ptr = static_cast<float*>(infer_result[1].data());
        const std::vector plate_color_vec(plate_color_ptr, plate_color_ptr + 5);
        int max_Index = argmax(plate_color_vec);
        const std::string plate_color = plate_color_list[max_Index];
        //车牌
        std::vector<int> plate_index;
        plate_index.reserve(21);
        auto* prob1_temp = static_cast<float*>(infer_result[0].data());
        for (size_t j = 0; j < 21; j++) {
            std::vector plate_tensor(prob1_temp, prob1_temp + 78);
            max_Index = argmax(plate_tensor);
            plate_index.push_back(max_Index);
            prob1_temp = prob1_temp + 78;
        }
        int pre = 0;
        std::string plate_str;
        for (const int j : plate_index) {
            if (j != 0 && j != pre) {
                plate_str += plate_chr[j];
            }
            pre = j;
        }

        result->car_plate_strs.emplace_back(plate_str);
        result->car_plate_colors.emplace_back(plate_color);
        return true;
    }

    bool LprRecognizer::predict(cv::Mat& image, LprResult* result) {
        std::vector<MDTensor> input_tensors(1);
        if (!preprocess(image, &input_tensors[0])) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].name = get_input_info(0).name;
        std::vector<MDTensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            std::cerr << "Failed to inference." << std::endl;
            return false;
        }
        postprocess(output_tensors, result);
        return true;
    }


} // namespace modeldeploy::vision::facedet

#include "csrc/vision/face_landmark/face_landmark.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"


namespace modeldeploy::vision::facealign {
    FaceLandmark1000::FaceLandmark1000(const std::string& model_file,
                                       const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool FaceLandmark1000::Initialize() {
        // parameters for preprocess
        size_ = {128, 128};

        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool FaceLandmark1000::Preprocess(
        cv::Mat* mat, MDTensor* output,
        std::map<std::string, std::array<int, 2>>* im_info) {
        // Resize
        int resize_w = size_[0];
        int resize_h = size_[1];
        if (resize_h != mat->rows || resize_w != mat->cols) {
            Resize::Run(mat, resize_w, resize_h);
        }

        // BRG2GRAY
        BGR2GRAY::Run(mat);

        // Record output shape of preprocessed image
        (*im_info)["output_shape"] = {mat->rows, mat->cols};
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");
        utils::mat_to_tensor(*mat, output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool FaceLandmark1000::Postprocess(
        MDTensor& infer_result, FaceAlignmentResult* result,
        const std::map<std::string, std::array<int, 2>>& im_info) {
        if (infer_result.shape[0] != 1) { std::cerr << "Only support batch = 1 now." << std::endl; };
        if (infer_result.dtype != MDDataType::Type::FP32) {
            std::cerr << "Only support post process with float32 data." << std::endl;
            return false;
        }

        auto iter_in = im_info.find("input_shape");
        if (iter_in == im_info.end()) { std::cerr << "Cannot find input_shape from im_info." << std::endl; };

        int in_h = iter_in->second[0];
        int in_w = iter_in->second[1];

        result->Clear();
        float* data = static_cast<float*>(infer_result.data());
        for (size_t i = 0; i < infer_result.shape[1]; i += 2) {
            float x = data[i];
            float y = data[i + 1];
            x = std::min(std::max(0.f, x), 1.0f);
            y = std::min(std::max(0.f, y), 1.0f);
            // decode landmarks (default 106 landmarks)
            result->landmarks.emplace_back(std::array<float, 2>{x * in_w, y * in_h});
        }

        return true;
    }

    bool FaceLandmark1000::Predict(cv::Mat* im, FaceAlignmentResult* result) {
        std::vector<MDTensor> input_tensors(1);

        std::map<std::string, std::array<int, 2>> im_info;

        // Record the shape of image and the shape of preprocessed image
        im_info["input_shape"] = {im->rows, im->cols};
        im_info["output_shape"] = {im->rows, im->cols};

        if (!Preprocess(im, &input_tensors[0], &im_info)) {
            std::cerr << "Failed to preprocess input image." << std::endl;
            return false;
        }
        input_tensors[0].name = get_input_info(0).name;
        std::vector<MDTensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            std::cerr << "Failed to inference." << std::endl;
            return false;
        }
        if (!Postprocess(output_tensors[0], result, im_info)) {
            std::cerr << "Failed to post process." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::facealign

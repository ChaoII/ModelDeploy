#include "csrc/vision/face_antispoof/face_anti_spoof_first.h"

#include <csrc/core/md_log.h>

#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/color_space_convert.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/convert.h"
#include "csrc/vision/common/processors/center_crop.h"
#include <variant>
#include <csrc/utils/utils.h>

namespace modeldeploy::vision::facedet {
    SeetaFaceAntiSpoofFirst::SeetaFaceAntiSpoofFirst(const std::string& model_file,
                                                     const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = Initialize();
    }

    bool SeetaFaceAntiSpoofFirst::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR("Failed to initialize fastdeploy backend.");
            return false;
        }
        return true;
    }

    bool SeetaFaceAntiSpoofFirst::preprocess(cv::Mat* mat, MDTensor* output) {
        Resize::Run(mat, 256, 256);
        CenterCrop::Run(mat, 224, 224);
        cv::cvtColor(*mat, *mat, cv::COLOR_BGR2YCrCb);
        HWC2CHW::Run(mat);
        Cast::Run(mat, "float");
        if (!utils::mat_to_tensor(*mat, output)) {
            MD_LOG_ERROR("Failed to binding mat to tensor.");
            return false;
        }
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }


    bool SeetaFaceAntiSpoofFirst::postprocess(const std::vector<MDTensor>& infer_result, float* result) {
        MDTensor infer_result_ = infer_result[0];
        infer_result_.squeeze();
        infer_result_.print_info();
        *result = static_cast<float*>(infer_result_.buffer_)[1];
        return true;
    }

    bool SeetaFaceAntiSpoofFirst::predict(cv::Mat* im, float* result) {
        std::vector<MDTensor> input_tensors(1);
        if (!preprocess(im, &input_tensors[0])) {
            MD_LOG_ERROR("Failed to preprocess input image.");
            return false;
        }

        input_tensors[0].name = get_input_info(0).name;
        std::vector<MDTensor> output_tensors;
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR("Failed to inference.");
            return false;
        }
        postprocess(output_tensors, result);
        return true;
    }
}

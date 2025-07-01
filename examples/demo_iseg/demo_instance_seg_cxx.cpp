//
// Created by aichao on 2025/4/14.
//

#include <iostream>
#include <filesystem>

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main(int argc, char** argv) {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_fp16 = true;
    // option.enable_trt = true;
    std::string model_file = "../../test_data/test_models/yolo11n-seg_nms.onnx";
    std::string image_file = "../../test_data/test_images/test_person.jpg";
    auto model = modeldeploy::vision::detection::UltralyticsSeg(model_file, option);
    auto im = cv::imread(image_file);
    TimerArray times;
    int loop = 100;
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    for (int i = 0; i < loop; i++) {
        model.predict(im, &res, &times);
    }
    times.print_benchmark();
    // res.display();
    auto vis_im = modeldeploy::vision::vis_iseg(im, res, 0.2, "../../test_data/test_models/font.ttf", 14, 0.5, 0);
    cv::imshow("vis_im", vis_im);
    cv::waitKey(0);
}

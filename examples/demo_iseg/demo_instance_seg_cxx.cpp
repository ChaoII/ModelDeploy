//
// Created by aichao on 2025/4/14.
//

#include <iostream>
#include <filesystem>
#include <csrc/core/md_log.h>

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main(int argc, char** argv) {
    modeldeploy::RuntimeOption option;
    // option.use_gpu();
    std::string model_file = "../../test_data/test_models/yolo11n-seg.onnx";
    auto model = modeldeploy::vision::detection::UltralyticsSeg(model_file, option);
    if (!model.is_initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return -1;
    }
    std::string image_file = "../../test_data/test_images/test_person.jpg";
    auto im = cv::imread(image_file);
    std::vector<modeldeploy::vision::InstanceSegResult> res;
    if (!model.predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    // res.display();
    auto vis_im = modeldeploy::vision::vis_iseg(im, res, 0.2, "../../test_data/test_models/font.ttf", 14, 0.5, 0);
    cv::resize(vis_im, vis_im, cv::Size(0, 0), 0.75, 0.75);
    cv::imshow("vis_im", vis_im);
    cv::waitKey(0);
}

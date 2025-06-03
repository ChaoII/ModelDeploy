//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include <string>
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    auto model = modeldeploy::vision::lpr::LprDetection("../../test_data/test_models/yolov5plate.onnx");

    auto im = cv::imread("../../test_data/test_images/test_lpr_pipeline2.jpg");
    auto im_bak = im.clone();

    std::vector<modeldeploy::vision::DetectionLandmarkResult> res;
    if (!model.predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    // res.display();
    const auto image = modeldeploy::vision::vis_det_landmarks(im_bak, res,
                                                              "../../test_data/msyh.ttc", 14, 2, 0.3, true);

    // cv::resize(image, image, cv::Size(0, 0), 0.5, 0.5);

    cv::imshow("result", image);
    cv::waitKey(0);
    return 0;
}

//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include <string>
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    auto model = modeldeploy::vision::facedet::CarPlateDetection("../../test_data/test_models/yolov5plate.onnx");
    if (!model.initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return -1;
    }
    auto im = cv::imread("../../test_data/test_images/test_carplate_detection.jpg");
    const auto im_bak = im.clone();

    modeldeploy::vision::DetectionLandmarkResult res;
    if (!model.predict(&im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    res.display();
    const auto image = VisFaceDetection(im_bak, res,
                                        "../../test_data/msyh.ttc", 14, 2, 0.3, true);
    cv::imshow("result", image);
    cv::waitKey(0);
    return 0;
}

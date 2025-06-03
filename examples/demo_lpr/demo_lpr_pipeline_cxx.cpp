//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

#ifdef WIN32
#include <windows.h>
#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    auto model = modeldeploy::vision::lpr::LprPipeline(
        "../../test_data/test_models/yolov5plate.onnx",
        "../../test_data/test_models/plate_recognition_color.onnx");
    auto im = cv::imread("../../test_data/test_images/test_lpr_pipeline3.jpg");
    auto im_bak = im.clone();
    std::vector<modeldeploy::vision::LprResult> res;
    if (!model.predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }

    auto vis_image = modeldeploy::vision::vis_lpr(im_bak, res, "../../test_data/msyh.ttc");
    cv::imshow("result", vis_image);
    cv::waitKey(0);
    return 0;
}

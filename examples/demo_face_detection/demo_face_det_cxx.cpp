//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    std::string image_file = "../../test_data/test_images/test_face_recognition.png";
    auto model =
        modeldeploy::vision::face::SCRFD("../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx");

    auto im = cv::imread(image_file);
    auto im_bak = im.clone();
    modeldeploy::vision::DetectionLandmarkResult res;
    if (!model.predict(&im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    res.display();
    auto image = VisFaceDetection(im_bak, res, "../../test_data/msyh.ttc", 14, 2, 0.3);
    cv::imshow("image", image);
    cv::waitKey(0);
    auto vis_im_list =
        modeldeploy::vision::utils::AlignFaceWithFivePoints(im_bak, res);
    if (!vis_im_list.empty()) {
        cv::imwrite("vis_result.jpg", vis_im_list[0]);
        auto img_crop = modeldeploy::vision::utils::center_crop(vis_im_list[0], {248, 248});
        cv::imshow("img_crop", img_crop);
        cv::waitKey(0);
        cv::imwrite("vis_result_crop.jpg", img_crop);
        std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
    }
    return 0;
}

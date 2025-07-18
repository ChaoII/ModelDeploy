//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    std::string image_file = "../../test_data/test_images/test_face_detection4.jpg";
    auto model =
        modeldeploy::vision::face::Scrfd(
            "../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx");
    auto im = modeldeploy::ImageData::imread(image_file);
    auto im_bak = im.clone();
    std::vector<modeldeploy::vision::DetectionLandmarkResult> res;

    int loop = 100;
    TimerArray timers;
    for (int i = 0; i < loop; i++) {
        if (!model.predict(im, &res, &timers)) {
            std::cerr << "Failed to predict." << std::endl;
            return -1;
        }
    }
    timers.print_benchmark();
    // res.display();
    const auto image = modeldeploy::vision::vis_det_landmarks(im_bak, res, "../../test_data/msyh.ttc", 14, 2, 0.3);
    image.imshow("result");
    const auto vis_im_list =
        modeldeploy::vision::utils::align_face_with_five_points(im, res);
    if (!vis_im_list.empty()) {
        std::vector<modeldeploy::ImageData> cropped_images;
        for (auto& align_image : vis_im_list) {
            auto img_crop = modeldeploy::vision::utils::center_crop(align_image, {248, 248});
            cropped_images.push_back(img_crop);
            img_crop.imshow("align_face");
        }
        // int size = static_cast<int>(std::sqrt(cropped_images.size())) + 1;
        // // 如果图像数量不足，填充空白图像
        // int total_images = size * size;
        // while (cropped_images.size() < total_images) {
        //     cv::Mat image(248, 248, CV_8UC3, cv::Scalar(0, 0, 0)); // 黑色空白图像
        //     cropped_images.push_back(blank_image);
        // }
        //
        // // 创建一个向量来存储每一行的拼接图像
        // std::vector<cv::Mat> rows;
        // // 水平拼接每一行
        // for (int i = 0; i < size; ++i) {
        //     std::vector<cv::Mat> row_images(cropped_images.begin() + i * size, cropped_images.begin() + (i + 1) * size);
        //     cv::Mat row;
        //     cv::hconcat(row_images, row);
        //     rows.push_back(row);
        // }
        //
        // // 垂直拼接所有行
        // cv::Mat concatenated_image;
        // cv::vconcat(rows, concatenated_image);
        //
        // // 显示拼接后的图像
        // cv::imshow("Concatenated Image", concatenated_image);
        // cv::waitKey(0);
    }
    return 0;
}

//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <string>
#include "csrc/vision.h"
#include "csrc/vision/utils.h"

std::vector<int> GenerateColorMap(int num_classes) {
    if (num_classes < 10) {
        num_classes = 10;
    }
    std::vector<int> color_map(num_classes * 3, 0);
    for (int i = 0; i < num_classes; ++i) {
        int j = 0;
        int lab = i;
        while (lab) {
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j));
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
            ++j;
            lab >>= 3;
        }
    }
    return color_map;
}

cv::Mat VisFaceDetection(const cv::Mat& im, const modeldeploy::vision::FaceDetectionResult& result,
                         int line_size, float font_size) {
    auto color_map = GenerateColorMap(10);
    int h = im.rows;
    int w = im.cols;

    auto vis_im = im.clone();
    bool vis_landmarks = false;
    if ((result.landmarks_per_face > 0) &&
        (result.boxes.size() * result.landmarks_per_face ==
            result.landmarks.size())) {
        vis_landmarks = true;
    }
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        cv::Rect rect(result.boxes[i][0], result.boxes[i][1],
                      result.boxes[i][2] - result.boxes[i][0],
                      result.boxes[i][3] - result.boxes[i][1]);
        int color_id = i % 333;
        int c0 = color_map[3 * color_id + 0];
        int c1 = color_map[3 * color_id + 1];
        int c2 = color_map[3 * color_id + 2];
        cv::Scalar rect_color = cv::Scalar(c0, c1, c2);
        std::string text = std::to_string(result.scores[i]);
        if (text.size() > 4) {
            text = text.substr(0, 4);
        }
        int font = cv::FONT_HERSHEY_SIMPLEX;
        cv::Size text_size = cv::getTextSize(text, font, font_size, 1, nullptr);
        cv::Point origin;
        origin.x = rect.x;
        origin.y = rect.y;
        cv::Rect text_background =
            cv::Rect(result.boxes[i][0], result.boxes[i][1] - text_size.height,
                     text_size.width, text_size.height);
        cv::rectangle(vis_im, rect, rect_color, line_size);
        cv::putText(vis_im, text, origin, font, font_size,
                    cv::Scalar(255, 255, 255), 1);
        // vis landmarks (if have)
        if (vis_landmarks) {
            cv::Scalar landmark_color = rect_color;
            for (size_t j = 0; j < result.landmarks_per_face; ++j) {
                cv::Point landmark;
                landmark.x = static_cast<int>(
                    result.landmarks[i * result.landmarks_per_face + j][0]);
                landmark.y = static_cast<int>(
                    result.landmarks[i * result.landmarks_per_face + j][1]);
                cv::circle(vis_im, landmark, line_size, landmark_color, -1);
            }
        }
    }
    return vis_im;
}


int main() {
    std::string image_file = "../../test_data/test_images/test_face_recognition.png";
    auto model = modeldeploy::vision::face::SCRFD("../../test_data/test_models/face/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!model.initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return -1;
    }

    auto im = cv::imread(image_file);
    auto im_bak = im.clone();
    modeldeploy::vision::FaceDetectionResult res;
    if (!model.predict(&im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << res.Str() << std::endl;
    auto image = VisFaceDetection(im_bak, res, 2, 0.5);
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

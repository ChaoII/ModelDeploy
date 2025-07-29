//
// Created by 84945 on 2025/6/2.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    struct PoseParams {
        static constexpr float kpt_threshold = 0.5;
        static constexpr bool is_draw_kpt_line = true;
        //If True, the function will draw lines connecting keypoint for human pose.Default is True.
        inline static auto person_color = cv::Scalar(0, 0, 255);
        inline static std::vector<std::vector<int>> skeleton = {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}
        };
        inline static std::vector<cv::Scalar> pose_palette =
        {
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 153, 51),
            cv::Scalar(255, 178, 102),
            cv::Scalar(230, 230, 0),
            cv::Scalar(255, 153, 255),
            cv::Scalar(153, 204, 255),
            cv::Scalar(255, 102, 255),
            cv::Scalar(255, 51, 255),
            cv::Scalar(102, 178, 255),
            cv::Scalar(51, 153, 255),
            cv::Scalar(255, 153, 153),
            cv::Scalar(255, 102, 102),
            cv::Scalar(255, 51, 51),
            cv::Scalar(153, 255, 153),
            cv::Scalar(102, 255, 102),
            cv::Scalar(51, 255, 51),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 0, 0),
            cv::Scalar(255, 255, 255),
        };
        inline static std::vector<int> limb_color = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
        inline static std::vector<int> kpt_color = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};
        inline static std::map<unsigned int, std::string> kpt_body_names{
            {0, "Nose"},
            {1, "left_eye"}, {2, "right_eye"},
            {3, "left_ear"}, {4, "right_ear"},
            {5, "left_shoulder"}, {6, "right_shoulder"},
            {7, "left_elbow"}, {8, "right_elbow"},
            {9, "left_wrist"}, {10, "right_wrist"},
            {11, "left_hip"}, {12, "right_hip"},
            {13, "left_knee"}, {14, "right_knee"},
            {15, "left_ankle"}, {16, "right_ankle"}
        };
    };


    void draw_keypoints(cv::Mat& cv_image,
                        const std::vector<cv::Point3f>& keypoints,
                        const int keypoint_radius,
                        const bool is_draw_kpt_line = true) {
        for (int j = 0; j < keypoints.size(); ++j) {
            const cv::Point3f keypoint = keypoints[j];
            if (keypoint.z < PoseParams::kpt_threshold)
                continue;
            cv::Scalar kptColor = PoseParams::pose_palette[PoseParams::kpt_color[j]];
            cv::circle(cv_image, cv::Point(keypoint.x, keypoint.y), keypoint_radius, kptColor, -1, 8);
        }

        if (is_draw_kpt_line) {
            for (int j = 0; j < PoseParams::skeleton.size(); ++j) {
                const auto kpt0 = keypoints[PoseParams::skeleton[j][0] - 1];
                const auto kpt1 = keypoints[PoseParams::skeleton[j][1] - 1];
                if (kpt0.z < PoseParams::kpt_threshold || kpt1.z < PoseParams::kpt_threshold)
                    continue;
                cv::Scalar kptColor = PoseParams::pose_palette[PoseParams::limb_color[j]];
                cv::line(cv_image, cv::Point(kpt0.x, kpt0.y), cv::Point(kpt1.x, kpt1.y), kptColor, 1, cv::LINE_AA);
            }
        }
    }

    ImageData vis_pose(ImageData& image, const std::vector<PoseResult>& result,
                       const std::string& font_path, const int font_size,
                       const int landmark_radius, const double alpha,
                       const bool save_result) {
        cv::Mat cv_image, overlay;
        image.to_mat(&cv_image);
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        static std::map<int, cv::Scalar_<int>> color_map; // ← 每类颜色只初始化一次
        // 绘制半透明部分（填充矩形）
        for (const auto& _result : result) {
            auto class_id = _result.label_id;
            if (color_map.find(class_id) == color_map.end()) {
                color_map[class_id] = get_random_color();
            }
            auto cv_color = color_map[class_id];
            const std::string text = "score: " + std::to_string(_result.score).substr(0, 4);
            draw_rectangle_and_text(overlay, utils::rect2f_to_cv_type(_result.box), text, cv_color, font, font_size, -1,
                                    false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (const auto& _result : result) {
            auto class_id = _result.label_id;
            auto cv_color = color_map[class_id];
            const std::string text = "score: " + std::to_string(_result.score).substr(0, 4);
            draw_rectangle_and_text(cv_image, utils::rect2f_to_cv_type(_result.box), text, cv_color,
                                    font, font_size, 1, true);
            std::vector<cv::Point3f> cv_keypoints;
            std::transform(_result.keypoints.begin(), _result.keypoints.end(),
                           std::back_inserter(cv_keypoints),
                           [](const Point3f& point) {
                               return utils::point3f_to_cv_type(point);
                           });
            draw_keypoints(cv_image, cv_keypoints, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save pose result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return image;
    }
}

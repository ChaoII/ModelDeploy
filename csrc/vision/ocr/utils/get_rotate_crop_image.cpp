//
// Created by aichao on 2025/2/21.
//

#include <opencv2/opencv.hpp>
#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    cv::Mat get_rotate_crop_image(const cv::Mat& src_image,
                                  const std::array<int, 8>& box) {
        cv::Mat image;
        src_image.copyTo(image);
        std::vector<std::vector<int>> points;
        for (int i = 0; i < 4; ++i) {
            std::vector<int> tmp;
            tmp.push_back(box[2 * i]);
            tmp.push_back(box[2 * i + 1]);
            points.push_back(tmp);
        }
        int x_collect[4] = {box[0], box[2], box[4], box[6]};
        int y_collect[4] = {box[1], box[3], box[5], box[7]};
        int left = *std::min_element(x_collect, x_collect + 4);
        int right = *std::max_element(x_collect, x_collect + 4);
        int top = *std::min_element(y_collect, y_collect + 4);
        int bottom = *std::max_element(y_collect, y_collect + 4);
        cv::Mat img_crop;
        image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);
        for (auto & point : points) {
            point[0] -= left;
            point[1] -= top;
        }

        int img_crop_width = static_cast<int>(sqrt(pow(points[0][0] - points[1][0], 2) +
            pow(points[0][1] - points[1][1], 2)));
        int img_crop_height = static_cast<int>(sqrt(pow(points[0][0] - points[3][0], 2) +
            pow(points[0][1] - points[3][1], 2)));

        cv::Point2f pts_std[4];
        pts_std[0] = cv::Point2f(0., 0.);
        pts_std[1] = cv::Point2f(static_cast<float>(img_crop_width), 0.);
        pts_std[2] = cv::Point2f(static_cast<float>(img_crop_width), static_cast<float>(img_crop_height));
        pts_std[3] = cv::Point2f(0.f, static_cast<float>(img_crop_height));

        cv::Point2f pointsf[4];
        pointsf[0] = cv::Point2f(static_cast<float>(points[0][0]), static_cast<float>(points[0][1]));
        pointsf[1] = cv::Point2f(static_cast<float>(points[1][0]), static_cast<float>(points[1][1]));
        pointsf[2] = cv::Point2f(static_cast<float>(points[2][0]), static_cast<float>(points[2][1]));
        pointsf[3] = cv::Point2f(static_cast<float>(points[3][0]), static_cast<float>(points[3][1]));
        cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);
        cv::Mat dst_img;
        cv::warpPerspective(img_crop, dst_img, M,
                            cv::Size(img_crop_width, img_crop_height),
                            cv::BORDER_REPLICATE);

        if (static_cast<float>(dst_img.rows) >= static_cast<float>(dst_img.cols) * 1.5) {
            auto srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
            cv::transpose(dst_img, srcCopy);
            cv::flip(srcCopy, srcCopy, 0);
            return srcCopy;
        }
        return dst_img;
    }
}

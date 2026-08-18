//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    ImageData vis_sem(ImageData& image, const SemSegResult& result,
                      const std::unordered_map<int, std::string>& label_map,
                      double alpha, bool save_result) {
        cv::Mat cv_image;
        image.asMat(&cv_image);
        if (result.shape.size() != 2) {
            MD_LOG_WARN << "vis_sem: invalid result shape." << std::endl;
            return image;
        }
        const int H = static_cast<int>(result.shape[0]);
        const int W = static_cast<int>(result.shape[1]);
        const size_t num = static_cast<size_t>(H) * W;
        if (result.labels.size() < num) {
            MD_LOG_WARN << "vis_sem: label buffer too small." << std::endl;
            return image;
        }
        // Cityscapes 标准调色板（此处为 BGR 顺序，匹配 OpenCV Mat 的 CV_8UC3 布局）
        static const std::vector<cv::Scalar> palette = {
            {128, 64, 128}, {232, 35, 244}, {70, 70, 70}, {156, 102, 102},
            {153, 153, 190}, {153, 153, 153}, {30, 170, 250}, {0, 220, 220},
            {35, 142, 107}, {152, 251, 152}, {180, 130, 70}, {60, 20, 220},
            {0, 0, 255}, {142, 0, 0}, {70, 0, 0}, {100, 60, 0},
            {100, 80, 0}, {230, 0, 0}, {32, 11, 119},
        };
        // 生成类别彩色图
        cv::Mat color_map(H, W, CV_8UC3);
        for (int y = 0; y < H; ++y) {
            const uint8_t* lrow = result.labels.data() + static_cast<size_t>(y) * W;
            cv::Vec3b* drow = color_map.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x) {
                const int cls = lrow[x];
                const cv::Scalar& c = (cls >= 0 && cls < static_cast<int>(palette.size()))
                    ? palette[cls] : cv::Scalar(0, 0, 0);
                drow[x] = cv::Vec3b(static_cast<uchar>(c[0]), static_cast<uchar>(c[1]), static_cast<uchar>(c[2]));
            }
        }
        // resize 到原图尺寸并叠加
        cv::Mat color_resized;
        if (color_map.size() != cv_image.size()) {
            cv::resize(color_map, color_resized, cv_image.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            color_resized = color_map;
        }
        cv::addWeighted(cv_image, 1.0 - alpha, color_resized, alpha, 0.0, cv_image);
        if (save_result) {
            cv::imwrite("vis_sem.jpg", cv_image);
        }
        return ImageData(cv_image);
    }
}

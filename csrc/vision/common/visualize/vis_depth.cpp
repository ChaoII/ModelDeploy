//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    ImageData vis_depth(ImageData& image, const DepthResult& result,
                        bool colorize, bool save_result) {
        cv::Mat cv_image;
        image.to_mat(cv_image);
        if (result.shape.size() != 2) {
            MD_LOG_WARN << "vis_depth: invalid result shape." << std::endl;
            return image;
        }
        const int H = static_cast<int>(result.shape[0]);
        const int W = static_cast<int>(result.shape[1]);
        const size_t num = static_cast<size_t>(H) * W;
        if (result.depth.size() < num) {
            MD_LOG_WARN << "vis_depth: depth buffer too small." << std::endl;
            return image;
        }
        // 深度 -> 归一化 0-1（min-max），转伪彩色
        float mn = result.depth[0], mx = result.depth[0];
        for (size_t i = 1; i < num; ++i) {
            mn = std::min(mn, result.depth[i]);
            mx = std::max(mx, result.depth[i]);
        }
        const float range = (mx - mn) > 1e-6f ? (mx - mn) : 1.0f;
        cv::Mat depth_f(H, W, CV_32FC1);
        for (int y = 0; y < H; ++y) {
            const float* srow = result.depth.data() + static_cast<size_t>(y) * W;
            float* drow = depth_f.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                drow[x] = (srow[x] - mn) / range;
            }
        }
        cv::Mat vis;
        if (colorize) {
            // applyColorMap 需要 8UC1 输入，先转 8UC1（0-255）
            cv::Mat depth8;
            depth_f.convertTo(depth8, CV_8UC1, 255.0);
            cv::applyColorMap(depth8, vis, cv::COLORMAP_JET);
        } else {
            depth_f.convertTo(vis, CV_8UC1, 255.0);
        }
        if (vis.size() != cv_image.size()) {
            cv::resize(vis, vis, cv_image.size(), 0, 0, cv::INTER_LINEAR);
        }
        if (save_result) {
            cv::imwrite("vis_depth.jpg", vis);
        }
        return ImageData(vis);
    }
}

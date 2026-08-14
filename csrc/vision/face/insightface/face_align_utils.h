//
// insightface 人脸对齐工具：与 python insightface.utils.face_align 逐值对齐。
//
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include "core/md_decl.h"

namespace modeldeploy::vision::face {

    // ArcFace 标准 5 点参考（112x112 空间）
    extern const std::array<std::array<float, 2>, 5> kArcfaceDst;

    // 构造 NCHW float32 blob：BGR 图 -> (mean 减/scale 除 可选 swapRB)，输出 [1,C,H,W]
    // 替代 cv::dnn::blobFromImage（OpenCV 5 预编译包无 dnn 模块）
    MODELDEPLOY_CXX_EXPORT cv::Mat make_blob_from_image(const cv::Mat& img,
                                                        double scale, const cv::Scalar& mean,
                                                        bool swap_rb);

    // 2x3 仿射矩阵求逆（替代 cv::invertAffineTransform，OpenCV 5 位置变动）
    MODELDEPLOY_CXX_EXPORT cv::Mat invert_affine_transform(const cv::Mat& M);

    // 估计相似变换矩阵（lmk -> dst），与 skimage SimilarityTransform.estimate 一致
    // 返回 2x3 仿射矩阵 M，cv::warpAffine 用
    MODELDEPLOY_CXX_EXPORT cv::Mat estimate_norm(const std::vector<std::array<float, 2>>& lmk,
                                                 int image_size = 112);

    // 对齐裁剪：warpAffine 到 image_size x image_size
    MODELDEPLOY_CXX_EXPORT cv::Mat norm_crop(const cv::Mat& img,
                                             const std::vector<std::array<float, 2>>& landmark,
                                             int image_size = 112);

    // 以 center 为中心 transform（landmark.py 的 face_align.transform），
    // 返回裁剪图 + 2x3 仿射矩阵 M（前向，warpAffine 用）
    MODELDEPLOY_CXX_EXPORT cv::Mat transform(const cv::Mat& data,
                                             const std::array<float, 2>& center,
                                             int output_size, float scale,
                                             float rotation, cv::Mat* M_out);

    // 用逆仿射矩阵把点映射回原图（trans_points）
    MODELDEPLOY_CXX_EXPORT void trans_points2d(std::vector<std::array<float, 2>>* pts,
                                               const cv::Mat& inv_M);
    MODELDEPLOY_CXX_EXPORT void trans_points3d(std::vector<std::array<float, 3>>* pts,
                                               const cv::Mat& inv_M);

} // namespace modeldeploy::vision::face

//
// insightface 人脸对齐工具实现：与 python insightface.utils.face_align 逐值对齐。
// 关键：skimage SimilarityTransform.estimate 用 _umeyama（Umeyama 算法）。
//
#include "vision/face/insightface/face_align_utils.h"
#include <cmath>
#include <cfloat>

namespace modeldeploy::vision::face {

    const std::array<std::array<float, 2>, 5> kArcfaceDst = {{
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    }};

    namespace {

        // 2x2 SVD（对称 A^T A 的 Jacobi 旋转 + LAPACK 符号约定，与 numpy.linalg.svd 对齐）
        // A = U * S * V^T，返回 U, s0, s1, V（V 为列向量，V 列主元非负）
        void svd2x2(double a00, double a01, double a10, double a11,
                    double* U00, double* U01, double* U10, double* U11,
                    double* s0, double* s1,
                    double* V00, double* V01, double* V10, double* V11) {
            const double AtA00 = a00 * a00 + a10 * a10;
            const double AtA01 = a00 * a01 + a10 * a11;
            const double AtA11 = a01 * a01 + a11 * a11;
            double t = 0.0;
            if (std::fabs(AtA01) > 0.0) {
                const double tau = (AtA11 - AtA00) / (2.0 * AtA01);
                t = (tau >= 0 ? 1.0 : -1.0) / (std::fabs(tau) + std::hypot(1.0, tau));
            }
            const double cr = 1.0 / std::sqrt(1.0 + t * t);
            const double sr = t * cr;
            // V = [[cr, sr], [-sr, cr]]
            double v00 = cr, v01 = sr, v10 = -sr, v11 = cr;
            // U = A V
            double u0_0 = a00 * v00 + a01 * v10;
            double u0_1 = a00 * v01 + a01 * v11;
            double u1_0 = a10 * v00 + a11 * v10;
            double u1_1 = a10 * v01 + a11 * v11;
            double s0v = std::hypot(u0_0, u1_0);
            double s1v = std::hypot(u0_1, u1_1);
            if (s0v > 1e-14) { u0_0 /= s0v; u1_0 /= s0v; }
            if (s1v > 1e-14) { u0_1 /= s1v; u1_1 /= s1v; }
            // 奇异值降序 + 同步交换列
            if (s0v < s1v) {
                std::swap(s0v, s1v);
                std::swap(u0_0, u0_1); std::swap(u1_0, u1_1);
                std::swap(v00, v01); std::swap(v10, v11);
            }
            // LAPACK 符号约定：V 每列首元素非负（V[0][j] >= 0），U 同步翻转
            if (v00 < 0) { v00 = -v00; v10 = -v10; u0_0 = -u0_0; u1_0 = -u1_0; }
            if (v01 < 0) { v01 = -v01; v11 = -v11; u0_1 = -u0_1; u1_1 = -u1_1; }
            *s0 = s0v; *s1 = s1v;
            *U00 = u0_0; *U01 = u0_1; *U10 = u1_0; *U11 = u1_1;
            *V00 = v00; *V01 = v01; *V10 = v10; *V11 = v11;
        }

        // skimage _umeyama：src(dst?) 返回 3x3 齐次相似变换参数
        // 注意 skimage 中 src 是输入点，dst 是目标点（estimate(lmk, dst)）
        cv::Mat umeyama(const std::vector<cv::Point2d>& src,
                        const std::vector<cv::Point2d>& dst) {
            const int num = static_cast<int>(src.size());
            const int dim = 2;
            cv::Point2d src_mean(0, 0), dst_mean(0, 0);
            for (int i = 0; i < num; ++i) {
                src_mean.x += src[i].x; src_mean.y += src[i].y;
                dst_mean.x += dst[i].x; dst_mean.y += dst[i].y;
            }
            src_mean.x /= num; src_mean.y /= num;
            dst_mean.x /= num; dst_mean.y /= num;
            // src_demean, dst_demean
            std::vector<cv::Point2d> sd(num), dd(num);
            for (int i = 0; i < num; ++i) {
                sd[i] = {src[i].x - src_mean.x, src[i].y - src_mean.y};
                dd[i] = {dst[i].x - dst_mean.x, dst[i].y - dst_mean.y};
            }
            // A = (dst_demean^T @ src_demean) / num  (2x2)
            double A00 = 0, A01 = 0, A10 = 0, A11 = 0;
            for (int i = 0; i < num; ++i) {
                A00 += dd[i].x * sd[i].x; A01 += dd[i].x * sd[i].y;
                A10 += dd[i].y * sd[i].x; A11 += dd[i].y * sd[i].y;
            }
            A00 /= num; A01 /= num; A10 /= num; A11 /= num;
            // d = ones; if det(A)<0: d[-1] = -1
            const double detA = A00 * A11 - A01 * A10;
            double d0 = 1.0, d1 = 1.0;
            if (detA < 0) d1 = -1.0;
            // SVD: A = U S V^T
            double U00, U01, U10, U11, s0, s1, V00, V01, V10, V11;
            svd2x2(A00, A01, A10, A11, &U00, &U01, &U10, &U11, &s0, &s1, &V00, &V01, &V10, &V11);
            // rank 判断（2x2 满秩）
            // T[:dim,:dim] = U @ diag(d) @ V^T
            // diag(d) @ V^T
            double DV00 = d0 * V00, DV01 = d0 * V10; // V^T 行0 = (V00, V10)
            double DV10 = d1 * V01, DV11 = d1 * V11; // V^T 行1 = (V01, V11)
            double R00 = U00 * DV00 + U01 * DV10;
            double R01 = U00 * DV01 + U01 * DV11;
            double R10 = U10 * DV00 + U11 * DV10;
            double R11 = U10 * DV01 + U11 * DV11;
            // estimate_scale = True
            // scale = 1 / (src_demean.var(axis=0).sum()) * (S @ d)
            double var_sum = 0;
            for (int i = 0; i < num; ++i) { var_sum += sd[i].x * sd[i].x + sd[i].y * sd[i].y; }
            var_sum /= num;
            const double scale = (s0 * d0 + s1 * d1) / var_sum;
            // T[:dim, dim] = dst_mean - scale * (R @ src_mean)
            // 2x3 仿射：x' = R*x*scale + t
            cv::Mat M = cv::Mat::eye(3, 3, CV_64F);
            M.at<double>(0, 0) = R00 * scale; M.at<double>(0, 1) = R01 * scale;
            M.at<double>(1, 0) = R10 * scale; M.at<double>(1, 1) = R11 * scale;
            M.at<double>(0, 2) = dst_mean.x - scale * (R00 * src_mean.x + R01 * src_mean.y);
            M.at<double>(1, 2) = dst_mean.y - scale * (R10 * src_mean.x + R11 * src_mean.y);
            return M;
        }
    } // namespace

    cv::Mat invert_affine_transform(const cv::Mat& M) {
        // M 为 2x3: [a b c; d e f]，逆为:
        // det = a*e - b*d; inv = [e/det, -b/det, (b*f - c*e)/det; -d/det, a/det, (c*d - a*f)/det]
        const double a = M.at<double>(0, 0), b = M.at<double>(0, 1), c = M.at<double>(0, 2);
        const double d = M.at<double>(1, 0), e = M.at<double>(1, 1), f = M.at<double>(1, 2);
        const double det = a * e - b * d;
        cv::Mat inv(2, 3, CV_64F);
        inv.at<double>(0, 0) = e / det;   inv.at<double>(0, 1) = -b / det; inv.at<double>(0, 2) = (b * f - c * e) / det;
        inv.at<double>(1, 0) = -d / det;  inv.at<double>(1, 1) = a / det;  inv.at<double>(1, 2) = (c * d - a * f) / det;
        return inv;
    }

    cv::Mat estimate_norm(const std::vector<std::array<float, 2>>& lmk, int image_size) {
        // dst = arcface_dst * ratio (+diff_x)
        std::vector<cv::Point2d> dst(5);
        if (image_size % 112 == 0) {
            const double ratio = static_cast<double>(image_size) / 112.0;
            for (int i = 0; i < 5; ++i) {
                dst[i] = {kArcfaceDst[i][0] * ratio, kArcfaceDst[i][1] * ratio};
            }
        } else {
            const double ratio = static_cast<double>(image_size) / 128.0;
            const double diff_x = 8.0 * ratio;
            for (int i = 0; i < 5; ++i) {
                dst[i] = {kArcfaceDst[i][0] * ratio + diff_x, kArcfaceDst[i][1] * ratio};
            }
        }
        std::vector<cv::Point2d> src(5);
        for (int i = 0; i < 5; ++i) src[i] = {lmk[i][0], lmk[i][1]};
        const cv::Mat T = umeyama(src, dst);
        return T(cv::Rect(0, 0, 3, 2)).clone(); // 2x3
    }

    cv::Mat norm_crop(const cv::Mat& img,
                      const std::vector<std::array<float, 2>>& landmark,
                      int image_size) {
        const cv::Mat M = estimate_norm(landmark, image_size);
        cv::Mat warped;
        cv::warpAffine(img, warped, M, cv::Size(image_size, image_size), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        return warped;
    }

    void trans_points2d(std::vector<std::array<float, 2>>* pts, const cv::Mat& inv_M) {
        for (auto& p : *pts) {
            const double x = p[0], y = p[1];
            const double nx = inv_M.at<double>(0, 0) * x + inv_M.at<double>(0, 1) * y + inv_M.at<double>(0, 2);
            const double ny = inv_M.at<double>(1, 0) * x + inv_M.at<double>(1, 1) * y + inv_M.at<double>(1, 2);
            p[0] = static_cast<float>(nx);
            p[1] = static_cast<float>(ny);
        }
    }

    void trans_points3d(std::vector<std::array<float, 3>>* pts, const cv::Mat& inv_M) {
        const double scale = std::sqrt(inv_M.at<double>(0, 0) * inv_M.at<double>(0, 0) +
                                       inv_M.at<double>(0, 1) * inv_M.at<double>(0, 1));
        for (auto& p : *pts) {
            const double x = p[0], y = p[1];
            const double nx = inv_M.at<double>(0, 0) * x + inv_M.at<double>(0, 1) * y + inv_M.at<double>(0, 2);
            const double ny = inv_M.at<double>(1, 0) * x + inv_M.at<double>(1, 1) * y + inv_M.at<double>(1, 2);
            p[0] = static_cast<float>(nx);
            p[1] = static_cast<float>(ny);
            p[2] = static_cast<float>(p[2] * scale);
        }
    }

} // namespace modeldeploy::vision::face

#include "vision/tracking/matching/kalman_filter.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace modeldeploy::vision::tracking {
    namespace {
        constexpr int kStateDim = 8;
        constexpr int kMeasDim = 4;
        constexpr double kStdWeightPosition = 1.0 / 20.0;
        constexpr double kStdWeightVelocity = 1.0 / 160.0;

        struct Mat {
            int rows{0};
            int cols{0};
            std::vector<double> data;

            Mat() = default;
            Mat(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}

            double& operator()(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
            double operator()(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }
        };

        Mat make_identity(int n) {
            Mat m(n, n);
            for (int i = 0; i < n; ++i) m(i, i) = 1.0;
            return m;
        }

        Mat make_diag(const std::vector<double>& d) {
            const int n = static_cast<int>(d.size());
            Mat m(n, n);
            for (int i = 0; i < n; ++i) m(i, i) = d[i];
            return m;
        }

        Mat mat_mul(const Mat& a, const Mat& b) {
            Mat r(a.rows, b.cols);
            for (int i = 0; i < a.rows; ++i) {
                for (int j = 0; j < b.cols; ++j) {
                    double acc = 0.0;
                    for (int k = 0; k < a.cols; ++k) acc += a(i, k) * b(k, j);
                    r(i, j) = acc;
                }
            }
            return r;
        }

        Mat mat_transpose(const Mat& a) {
            Mat t(a.cols, a.rows);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) t(j, i) = a(i, j);
            return t;
        }

        Mat mat_add(const Mat& a, const Mat& b) {
            Mat r(a.rows, a.cols);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) r(i, j) = a(i, j) + b(i, j);
            return r;
        }

        Mat mat_sub(const Mat& a, const Mat& b) {
            Mat r(a.rows, a.cols);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) r(i, j) = a(i, j) - b(i, j);
            return r;
        }

        Mat mat_scale(const Mat& a, double s) {
            Mat r = a;
            for (auto& v : r.data) v *= s;
            return r;
        }

        // Inverse via Gauss-Jordan with partial pivoting. Returns empty on singular.
        Mat mat_inverse(const Mat& a) {
            const int n = a.rows;
            Mat aug(n, 2 * n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) aug(i, j) = a(i, j);
                aug(i, n + i) = 1.0;
            }
            for (int col = 0; col < n; ++col) {
                int piv = col;
                for (int r = col + 1; r < n; ++r) {
                    if (std::abs(aug(r, col)) > std::abs(aug(piv, col))) piv = r;
                }
                if (std::abs(aug(piv, col)) < 1e-12) return Mat();
                if (piv != col) {
                    for (int j = 0; j < 2 * n; ++j) std::swap(aug(col, j), aug(piv, j));
                }
                const double d = aug(col, col);
                for (int j = 0; j < 2 * n; ++j) aug(col, j) /= d;
                for (int r = 0; r < n; ++r) {
                    if (r == col) continue;
                    const double f = aug(r, col);
                    if (f == 0.0) continue;
                    for (int j = 0; j < 2 * n; ++j) aug(r, j) -= f * aug(col, j);
                }
            }
            Mat inv(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) inv(i, j) = aug(i, n + j);
            return inv;
        }

        Mat to_mat(const std::array<double, 8>& mean) {
            Mat m(kStateDim, 1);
            for (int i = 0; i < kStateDim; ++i) m(i, 0) = mean[i];
            return m;
        }

        std::array<double, 8> from_mat(const Mat& m) {
            std::array<double, 8> out{};
            for (int i = 0; i < kStateDim; ++i) out[i] = m(i, 0);
            return out;
        }

        Mat cov_to_mat(const std::array<std::array<double, 8>, 8>& cov) {
            Mat m(kStateDim, kStateDim);
            for (int i = 0; i < kStateDim; ++i)
                for (int j = 0; j < kStateDim; ++j) m(i, j) = cov[i][j];
            return m;
        }

        std::array<std::array<double, 8>, 8> cov_from_mat(const Mat& m) {
            std::array<std::array<double, 8>, 8> out{};
            for (int i = 0; i < kStateDim; ++i)
                for (int j = 0; j < kStateDim; ++j) out[i][j] = m(i, j);
            return out;
        }
    }  // namespace

    KalmanFilter::KalmanFilter() {
        for (auto& row : covariance_) row.fill(0.0);
    }

    std::array<double, 4> KalmanFilter::convert_rect_to_xyah(const Rect2f& box) {
        const double cx = box.x + box.width * 0.5;
        const double cy = box.y + box.height * 0.5;
        const double h = box.height;
        const double a = (h > 0.0) ? box.width / h : 0.0;
        return {cx, cy, a, h};
    }

    Rect2f KalmanFilter::convert_xyah_to_rect(const std::array<double, 4>& xyah) {
        const double h = xyah[3];
        const double w = xyah[2] * h;
        const double x = xyah[0] - w * 0.5;
        const double y = xyah[1] - h * 0.5;
        return {static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(w), static_cast<float>(h)};
    }

    void KalmanFilter::init(const Rect2f& box) {
        const auto z = convert_rect_to_xyah(box);
        for (int i = 0; i < kMeasDim; ++i) mean_[i] = z[i];
        for (int i = kMeasDim; i < kStateDim; ++i) mean_[i] = 0.0;

        const double h = z[3];
        const std::vector<double> std = {
            2.0 * kStdWeightPosition * h, 2.0 * kStdWeightPosition * h,
            1e-2, 2.0 * kStdWeightPosition * h,
            10.0 * kStdWeightVelocity * h, 10.0 * kStdWeightVelocity * h,
            1e-5, 10.0 * kStdWeightVelocity * h};
        for (int i = 0; i < kStateDim; ++i) {
            const double v = std[i] * std[i];
            for (auto& row : covariance_) row.fill(0.0);
            covariance_[i][i] = v;
        }
        initialized_ = true;
    }

    void KalmanFilter::predict() {
        if (!initialized_) return;
        const double h = mean_[3];
        const std::vector<double> std_pos = {
            kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h};
        const std::vector<double> std_vel = {
            kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h};
        std::vector<double> diag;
        diag.reserve(kStateDim);
        for (int i = 0; i < kMeasDim; ++i) diag.push_back(std_pos[i] * std_pos[i]);
        for (int i = 0; i < kMeasDim; ++i) diag.push_back(std_vel[i] * std_vel[i]);
        const Mat Q = make_diag(diag);

        Mat F(kStateDim, kStateDim);
        for (int i = 0; i < kStateDim; ++i) F(i, i) = 1.0;
        for (int i = 0; i < kMeasDim; ++i) F(i, i + kMeasDim) = 1.0;

        const Mat mean = to_mat(mean_);
        const Mat cov = cov_to_mat(covariance_);
        const Mat ft = mat_transpose(F);

        mean_ = from_mat(mat_mul(F, mean));
        covariance_ = cov_from_mat(mat_add(mat_mul(mat_mul(F, cov), ft), Q));
    }

    void KalmanFilter::update(const Rect2f& box) {
        if (!initialized_) return;
        const auto xyah = convert_rect_to_xyah(box);
        Mat z(kMeasDim, 1);
        for (int i = 0; i < kMeasDim; ++i) z(i, 0) = xyah[i];

        const double h = mean_[3];
        const std::vector<double> std_pos = {
            kStdWeightPosition * h, kStdWeightPosition * h, 1e-1, kStdWeightPosition * h};
        std::vector<double> diag;
        for (int i = 0; i < kMeasDim; ++i) diag.push_back(std_pos[i] * std_pos[i]);
        const Mat R = make_diag(diag);

        Mat H(kMeasDim, kStateDim);
        for (int i = 0; i < kMeasDim; ++i) H(i, i) = 1.0;

        const Mat mean = to_mat(mean_);
        const Mat cov = cov_to_mat(covariance_);
        const Mat ht = mat_transpose(H);

        const Mat s = mat_add(mat_mul(mat_mul(H, cov), ht), R);      // S = H P H^T + R
        const Mat sinv = mat_inverse(s);
        const Mat k = mat_mul(mat_mul(cov, ht), sinv);               // K = P H^T S^-1

        const Mat proj_mean = mat_mul(H, mean);
        const Mat innovation = mat_sub(z, proj_mean);                // z - H mean
        const Mat k_innov = mat_mul(k, innovation);

        mean_ = from_mat(mat_add(mean, k_innov));
        const Mat kcov = mat_mul(k, mat_mul(s, mat_transpose(k)));
        covariance_ = cov_from_mat(mat_sub(cov, kcov));
    }

    Rect2f KalmanFilter::get_state() const {
        if (!initialized_) {
            return {};
        }
        const std::array<double, 4> xyah = {mean_[0], mean_[1], mean_[2], mean_[3]};
        return convert_xyah_to_rect(xyah);
    }
}

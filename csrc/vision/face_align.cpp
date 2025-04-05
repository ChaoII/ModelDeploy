#include "csrc/vision/utils.h"
#include "csrc/core/md_log.h"

namespace modeldeploy::vision::utils {
    cv::Mat mean_axis0(const cv::Mat &src) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat output(1, dim, CV_32F);
        for (int i = 0; i < dim; i++) {
            float sum = 0;
            for (int j = 0; j < num; j++) {
                sum += src.at<float>(j, i);
            }
            output.at<float>(0, i) = sum / num;
        }
        return output;
    }

    cv::Mat elementwise_minus(const cv::Mat &A, const cv::Mat &B) {
        cv::Mat output(A.rows, A.cols, A.type());
        assert(B.cols == A.cols);
        if (B.cols == A.cols) {
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < B.cols; j++) {
                    output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
                }
            }
        }
        return output;
    }

    cv::Mat VarAxis0(const cv::Mat &src) {
        cv::Mat temp_ = elementwise_minus(src, mean_axis0(src));
        cv::multiply(temp_, temp_, temp_);
        return mean_axis0(temp_);
    }

    int MatrixRank(cv::Mat M) {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b non_zero_singular_values = w > 0.0001;
        int rank = countNonZero(non_zero_singular_values);
        return rank;
    }

    cv::Mat similar_transform(cv::Mat &dst, cv::Mat &src) {
        int num = dst.rows;
        int dim = dst.cols;
        cv::Mat src_mean = mean_axis0(dst);
        cv::Mat dst_mean = mean_axis0(src);
        cv::Mat src_demean = elementwise_minus(dst, src_mean);
        cv::Mat dst_demean = elementwise_minus(src, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;
        }
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
        cv::SVD::compute(A, S, U, V);
        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);
        } else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            } else {
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_ * V; // np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U * twp;
                d.at<float>(dim - 1, 0) = s;
            }
        } else {
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V.t(); // np.dot(np.diag(d), V.T)
            cv::Mat res = U * twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
        }
        cv::Mat var_ = VarAxis0(src_demean);
        double val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d, S, res);
        double scale = 1.0 / val * cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) =
                -T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat temp2 = src_mean.t(); // src_mean.T
        cv::Mat temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale * temp3;
        T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
    }

    std::vector<cv::Mat> align_face_with_five_points(
            cv::Mat &image, DetectionLandmarkResult &result,
            std::vector<std::array<float, 2>> std_landmarks,
            std::array<int, 2> output_size) {
        if (std_landmarks.size() != 5) {
            MD_LOG_ERROR << "The landmarks.size() must be 5." << std::endl;
        }
        if (image.empty()) {
            MD_LOG_ERROR << "The input_image can't be empty.";
        }
        std::vector<cv::Mat> output_images;
        output_images.reserve(result.scores.size());
        if (result.boxes.empty()) {
            MD_LOG_ERROR << "The result is empty." << std::endl;
            return output_images;
        }

        cv::Mat src(5, 2, CV_32FC1, std_landmarks.data());
        for (int i = 0; i < result.landmarks.size(); i += 5) {
            cv::Mat dst(5, 2, CV_32FC1, result.landmarks.data() + i);
            cv::Mat m = similar_transform(dst, src);
            cv::Mat map_matrix;
            cv::Rect map_matrix_r = cv::Rect(0, 0, 3, 2);
            cv::Mat(m, map_matrix_r).copyTo(map_matrix);
            cv::Mat cropped_image_aligned;
            cv::warpAffine(image, cropped_image_aligned, map_matrix,
                           {output_size[0], output_size[1]});
            if (cropped_image_aligned.empty()) {
                MD_LOG_ERROR << "croppedImageAligned is empty." << std::endl;
            }
            output_images.emplace_back(cropped_image_aligned);
        }
        return output_images;
    }
}

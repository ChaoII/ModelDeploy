//
// insightface buffalo_l landmark 实现：2d106det + 1k3d68。
// 前处理 fused_preprocess center-crop；后处理逆仿射 + 3D 姿态。
// 与 python insightface.model_zoo.landmark.Landmark 逐值对齐。
//
#include "core/md_log.h"
#include "vision/face/insightface/insightface_landmark.h"
#include "core/tensor.h"
#include <cmath>
#include <cstring>

namespace modeldeploy::vision::face {

    // meanshape_68.pkl（68x3 float32）
    static const float kMeanShape68[68][3] = {
        {-0.62669504f, -0.29269969f, -0.31400186f}, {-0.59966493f, -0.12250272f, -0.29244095f},
        {-0.57102609f, 0.05118740f, -0.25494650f}, {-0.53385669f, 0.21284838f, -0.18663338f},
        {-0.47973323f, 0.35064596f, -0.04769993f}, {-0.39575565f, 0.44651166f, 0.07328663f},
        {-0.29880843f, 0.51066059f, 0.17977794f}, {-0.18838681f, 0.55444044f, 0.31666005f},
        {0.00147083f, 0.58443922f, 0.38841578f}, {0.19099054f, 0.55170709f, 0.31433114f},
        {0.32692856f, 0.48957568f, 0.16839953f}, {0.44002613f, 0.40235800f, 0.03596243f},
        {0.50687873f, 0.31162494f, -0.09476063f}, {0.54089463f, 0.20452619f, -0.20267142f},
        {0.57411844f, 0.04570330f, -0.28417650f}, {0.59914166f, -0.14585932f, -0.29649565f},
        {0.62754363f, -0.30774805f, -0.30199483f}, {-0.47466812f, -0.43760464f, 0.23648711f},
        {-0.41665992f, -0.47175601f, 0.31599256f}, {-0.34753880f, -0.48407802f, 0.36611354f},
        {-0.26063988f, -0.47639099f, 0.39923173f}, {-0.16712302f, -0.45777544f, 0.41660920f},
        {0.12317404f, -0.45874012f, 0.42506194f}, {0.20636158f, -0.48041934f, 0.41578391f},
        {0.28667304f, -0.49014941f, 0.39195377f}, {0.36239693f, -0.47686017f, 0.35278630f},
        {0.42556816f, -0.45005804f, 0.29531878f}, {-0.00762794f, -0.32308865f, 0.46194378f},
        {-0.00787685f, -0.25573877f, 0.51046944f}, {-0.00768755f, -0.19917990f, 0.55254567f},
        {-0.00734512f, -0.14261691f, 0.59866673f}, {-0.14496188f, 0.03312694f, 0.41969100f},
        {-0.08434254f, 0.03127361f, 0.47317410f}, {-0.00549969f, 0.03975145f, 0.51466960f},
        {0.06346293f, 0.04613516f, 0.47922406f}, {0.13398391f, 0.02204026f, 0.41907868f},
        {-0.38675082f, -0.31339756f, 0.25963187f}, {-0.31666079f, -0.35007161f, 0.32852700f},
        {-0.23413791f, -0.35491428f, 0.33349338f}, {-0.15516235f, -0.31524932f, 0.31432810f},
        {-0.23092176f, -0.28427041f, 0.32558286f}, {-0.31750903f, -0.28516537f, 0.30988252f},
        {0.13895708f, -0.30982405f, 0.31828359f}, {0.21945593f, -0.35319215f, 0.33802760f},
        {0.30174652f, -0.34966511f, 0.33310229f}, {0.37665316f, -0.31351814f, 0.26322857f},
        {0.29669479f, -0.28714398f, 0.32201305f}, {0.21462442f, -0.29052764f, 0.33124179f},
        {-0.20143844f, 0.23736143f, 0.37953663f}, {-0.13732077f, 0.18578564f, 0.46525246f},
        {-0.07648587f, 0.15119343f, 0.50357169f}, {-0.00253589f, 0.16872700f, 0.51643765f},
        {0.06442103f, 0.15088020f, 0.50452411f}, {0.12646531f, 0.17947596f, 0.46859166f},
        {0.21824785f, 0.23899227f, 0.37567368f}, {0.13288260f, 0.28392839f, 0.44005090f},
        {0.06802233f, 0.29735431f, 0.47740418f}, {-0.00046907f, 0.30004069f, 0.48710942f},
        {-0.06934267f, 0.29696861f, 0.48054048f}, {-0.14252016f, 0.27430332f, 0.43808019f},
        {-0.17813474f, 0.23059049f, 0.39635897f}, {-0.07403064f, 0.21471879f, 0.46532628f},
        {-0.00263622f, 0.21414155f, 0.48322961f}, {0.05981617f, 0.21076396f, 0.47224411f},
        {0.16690002f, 0.23127100f, 0.39678901f}, {0.05980928f, 0.22376429f, 0.46641338f},
        {-0.00143436f, 0.22575909f, 0.47524396f}, {-0.07522077f, 0.23065609f, 0.46714753f},
    };

    namespace {
        // estimate_affine_matrix_3d23d + P2sRt + matrix2angle（与 python transform.py 一致）
        void estimate_affine_matrix_3d23d(const float X[68][3], const float Y[68][3], float P[3][4]) {
            double A[4][4] = {0};
            double B[4][3] = {0};
            for (int i = 0; i < 68; ++i) {
                const double h[4] = {X[i][0], X[i][1], X[i][2], 1.0};
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) A[r][c] += h[r] * h[c];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 3; ++c) B[r][c] += h[r] * Y[i][c];
            }
            double aug[4][7];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) aug[r][c] = A[r][c];
                for (int c = 0; c < 3; ++c) aug[r][4 + c] = B[r][c];
            }
            for (int col = 0; col < 4; ++col) {
                int piv = col;
                for (int r = col + 1; r < 4; ++r)
                    if (std::fabs(aug[r][col]) > std::fabs(aug[piv][col])) piv = r;
                if (piv != col)
                    for (int c = 0; c < 7; ++c) std::swap(aug[col][c], aug[piv][c]);
                const double pv = aug[col][col];
                if (std::fabs(pv) < 1e-12) continue;
                for (int c = 0; c < 7; ++c) aug[col][c] /= pv;
                for (int r = 0; r < 4; ++r) {
                    if (r == col) continue;
                    const double f = aug[r][col];
                    for (int c = 0; c < 7; ++c) aug[r][c] -= f * aug[col][c];
                }
            }
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 4; ++r) P[c][r] = static_cast<float>(aug[r][4 + c]);
        }

        void P2sRt(const float P[3][4], float* s, float R[3][3], float t[3]) {
            t[0] = P[0][3]; t[1] = P[1][3]; t[2] = P[2][3];
            const double nR1 = std::hypot(P[0][0], P[0][1], P[0][2]);
            const double nR2 = std::hypot(P[1][0], P[1][1], P[1][2]);
            *s = static_cast<float>((nR1 + nR2) / 2.0);
            double r1[3] = {P[0][0] / nR1, P[0][1] / nR1, P[0][2] / nR1};
            double r2[3] = {P[1][0] / nR2, P[1][1] / nR2, P[1][2] / nR2};
            double r3[3] = {r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]};
            for (int i = 0; i < 3; ++i) { R[0][i] = static_cast<float>(r1[i]); R[1][i] = static_cast<float>(r2[i]); R[2][i] = static_cast<float>(r3[i]); }
        }

        void matrix2angle(const float R[3][3], float pose[3]) {
            const double sy = std::sqrt(static_cast<double>(R[0][0]) * R[0][0] + R[1][0] * R[1][0]);
            const bool singular = sy < 1e-6;
            double x, y, z;
            if (!singular) {
                x = std::atan2(R[2][1], R[2][2]);
                y = std::atan2(-R[2][0], sy);
                z = std::atan2(R[1][0], R[0][0]);
            } else {
                x = std::atan2(-R[1][2], R[1][1]);
                y = std::atan2(-R[2][0], sy);
                z = 0;
            }
            pose[0] = static_cast<float>(x * 180.0 / CV_PI);
            pose[1] = static_cast<float>(y * 180.0 / CV_PI);
            pose[2] = static_cast<float>(z * 180.0 / CV_PI);
        }
    } // namespace

    // ==================== Preprocessor ====================

    InsightFaceLandmarkPreprocessor::InsightFaceLandmarkPreprocessor() {
        size_ = {192, 192};
    }

    bool InsightFaceLandmarkPreprocessor::run(const ImageData& image, const std::array<float, 4>& bbox,
                                              cv::Mat* M, Tensor* output) const {
        const int dst = size_[0];
        const float w = bbox[2] - bbox[0];
        const float h = bbox[3] - bbox[1];
        const float cx = (bbox[2] + bbox[0]) / 2.0f;
        const float cy = (bbox[3] + bbox[1]) / 2.0f;
        const float scale = static_cast<float>(dst) / (std::max(w, h) * 1.5f);
        // python transform（rotation=0）: dst = (src - center)*scale + dst/2
        // 2x3 仿射矩阵：x' = scale*x + t
        cv::Mat M2x3 = cv::Mat::zeros(2, 3, CV_64F);
        M2x3.at<double>(0, 0) = scale;
        M2x3.at<double>(1, 1) = scale;
        M2x3.at<double>(0, 2) = (dst / 2.0) - cx * scale;
        M2x3.at<double>(1, 2) = (dst / 2.0) - cy * scale;
        // 双线性 warpAffine（与 python cv2.warpAffine INTER_LINEAR 一致）
        cv::Mat src_mat;
        image.to_mat(src_mat);
        cv::Mat warped;
        cv::warpAffine(src_mat, warped, M2x3, cv::Size(dst, dst),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        // 模型输入 0-255 RGB（内部归一化），仅 swapRB；pad=0
        std::vector<float> blob(static_cast<size_t>(3) * dst * dst);
        const uint8_t* src = warped.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB
            float* plane = blob.data() + static_cast<size_t>(c) * dst * dst;
            for (int i = 0; i < dst * dst; ++i) plane[i] = static_cast<float>(src[i * 3 + src_c]);
        }
        output->allocate({1, 3, dst, dst}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        if (M) *M = M2x3;
        return true;
    }

    // ==================== Postprocessor ====================

    bool InsightFaceLandmarkPostprocessor::run_2d(const std::vector<Tensor>& infer_results,
                                                  const cv::Mat& inv_M, const int input_size,
                                                  std::vector<std::array<float, 2>>* landmarks) {
        const float* pred = static_cast<const float*>(infer_results[0].data());
        const int total = static_cast<int>(infer_results[0].size());
        const int n_pts = total / 2;
        const int lmk_num = 106;
        const int start = std::max(0, n_pts - lmk_num);
        const float scale_v = input_size / 2.0f;
        landmarks->resize(lmk_num);
        for (int i = 0; i < lmk_num; ++i) {
            const int idx = start + i;
            (*landmarks)[i] = {(pred[idx * 2] + 1.0f) * scale_v, (pred[idx * 2 + 1] + 1.0f) * scale_v};
        }
        trans_points2d(landmarks, inv_M);
        return true;
    }

    bool InsightFaceLandmarkPostprocessor::run_3d(const std::vector<Tensor>& infer_results,
                                                  const cv::Mat& inv_M, const int input_size,
                                                  std::vector<std::array<float, 3>>* landmarks,
                                                  std::array<float, 3>* pose) {
        const float* pred = static_cast<const float*>(infer_results[0].data());
        const int total = static_cast<int>(infer_results[0].size());
        const int n_pts = total / 3;
        const int lmk_num = 68;
        const int start = std::max(0, n_pts - lmk_num);
        const float scale_v = input_size / 2.0f;
        std::vector<std::array<float, 3>> pts(lmk_num);
        std::vector<std::array<float, 3>> pred_pts(lmk_num);
        for (int i = 0; i < lmk_num; ++i) {
            const int idx = start + i;
            const std::array<float, 3> p = {
                (pred[idx * 3] + 1.0f) * scale_v,
                (pred[idx * 3 + 1] + 1.0f) * scale_v,
                pred[idx * 3 + 2] * scale_v};
            pred_pts[i] = p;
            pts[i] = p;
        }
        // 逆仿射回原图（3D：z 乘 scale）
        trans_points3d(&pts, inv_M);
        *landmarks = std::move(pts);
        if (pose) {
            // 姿态用 transform 前 pred（与 python 一致）
            float P[3][4];
            float X[68][3], Y[68][3];
            for (int i = 0; i < 68; ++i) {
                X[i][0] = kMeanShape68[i][0]; X[i][1] = kMeanShape68[i][1]; X[i][2] = kMeanShape68[i][2];
                Y[i][0] = pred_pts[i][0]; Y[i][1] = pred_pts[i][1]; Y[i][2] = pred_pts[i][2];
            }
            estimate_affine_matrix_3d23d(X, Y, P);
            float s, R[3][3], t[3];
            P2sRt(P, &s, R, t);
            float pose_out[3];
            matrix2angle(R, pose_out);
            (*pose)[0] = pose_out[0]; (*pose)[1] = pose_out[1]; (*pose)[2] = pose_out[2];
        }
        return true;
    }

    // ==================== Model ====================

    InsightFaceLandmark::InsightFaceLandmark(const std::string& model_file,
                                             const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceLandmark::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceLandmark::predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                                            std::vector<std::array<float, 2>>* landmarks,
                                            TimerArray* timers) {
        if (!landmarks) return false;
        cv::Mat M;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, bbox, &M, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_2d(reused_output_tensors_, inv_M, input_size_[0], landmarks)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    bool InsightFaceLandmark::predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                                           std::vector<std::array<float, 3>>* landmarks,
                                           std::array<float, 3>* pose,
                                           TimerArray* timers) {
        if (!landmarks) return false;
        cv::Mat M;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, bbox, &M, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_3d(reused_output_tensors_, inv_M, input_size_[0], landmarks, pose)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceLandmark> InsightFaceLandmark::clone() const {
        auto clone_model = std::make_unique<InsightFaceLandmark>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face

//
// Created by aichao on 2025/3/26.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/common/processors/resize.h"
#include "vision/common/processors/hwc2chw.h"
#include "vision/face/face_as/face_as_pipeline.h"


namespace modeldeploy::vision::face {
    SeetaFaceAsPipeline::SeetaFaceAsPipeline(
        const std::string& face_det_model_file,
        const std::string& first_model_file,
        const std::string& second_model_file,
        const RuntimeOption& option) {
        face_det_ = std::make_unique<Scrfd>(face_det_model_file, option);
        face_as_first_ = std::make_unique<SeetaFaceAsFirst>(first_model_file, option);
        face_as_second_ = std::make_unique<SeetaFaceAsSecond>(second_model_file, option);
    }


    bool SeetaFaceAsPipeline::is_initialized() const {
        if (face_det_ != nullptr && !face_det_->is_initialized()) {
            return false;
        }
        if (face_as_first_ != nullptr && !face_as_first_->is_initialized()) {
            return false;
        }
        if (face_as_second_ != nullptr && !face_as_second_->is_initialized()) {
            return false;
        }
        return true;
    }

    static float re_blur(const unsigned char* data, int width, int height) {
        float blur_val = 0.0;
        constexpr float kernel[9] = {
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 9.0
        };
        std::vector<float> BVer(width * height);
        std::vector<float> BHor(width * height);
        float filter_data = 0.0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (i < 4 || i > height - 5) {
                    BVer[i * width + j] = data[i * width + j];
                }
                else {
                    filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] +
                        kernel[2] * data[(i - 2) * width + j] +
                        kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i) * width + j] +
                        kernel[5] * data[(i + 1) * width + j] +
                        kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] +
                        kernel[8] * data[(i + 4) * width + j];
                    BVer[i * width + j] = filter_data;
                }

                if (j < 4 || j > width - 5) {
                    BHor[i * width + j] = data[i * width + j];
                }
                else {
                    filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] +
                        kernel[2] * data[i * width + (j - 2)] +
                        kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] +
                        kernel[5] * data[i * width + (j + 1)] +
                        kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] +
                        kernel[8] * data[i * width + (j + 4)];
                    BHor[i * width + j] = filter_data;
                }
            }
        }


        float D_Fver = 0.0;
        float D_FHor = 0.0;
        float D_BVer = 0.0;
        float D_BHor = 0.0;
        float s_FVer = 0.0;
        float s_FHor = 0.0;
        float s_Vver = 0.0;
        float s_VHor = 0.0;
        for (int i = 1; i < height; ++i) {
            for (int j = 1; j < width; ++j) {
                D_Fver = std::abs(
                    static_cast<float>(data[i * width + j]) - static_cast<float>(data[(i - 1) * width + j]));
                s_FVer += D_Fver;
                D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
                s_Vver += std::max(static_cast<float>(0.0), D_Fver - D_BVer);
                D_FHor = std::abs(
                    static_cast<float>(data[i * width + j]) - static_cast<float>(data[i * width + (j - 1)]));
                s_FHor += D_FHor;
                D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
                s_VHor += std::max(static_cast<float>(0.0), D_FHor - D_BHor);
            }
        }
        float b_FVer = (s_FVer - s_Vver) / s_FVer;
        float b_FHor = (s_FHor - s_VHor) / s_FHor;
        blur_val = std::max(b_FVer, b_FHor);
        return blur_val;
    }

    float clarity_estimate(cv::Mat& image) {
        if (!image.data || image.cols < 9 || image.rows < 9) return 0.0;
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        float blur_val = re_blur(image.data, image.cols, image.rows);
        float clarity = 1.0f - blur_val;
        float T1 = 0.3;
        float T2 = 0.55;
        if (clarity <= T1) {
            clarity = 0.0;
        }
        else if (clarity >= T2) {
            clarity = 1.0;
        }
        else {
            clarity = (clarity - T1) / (T2 - T1);
        }
        return clarity;
    }


    bool SeetaFaceAsPipeline::predict(const cv::Mat& im,
                                      std::vector<FaceAntiSpoofResult>* results,
                                      const float fuse_threshold,
                                      const float clarity_threshold) const {
        cv::Mat im_bak0 = im.clone();
        cv::Mat im_bak1 = im.clone();
        std::vector<std::tuple<int, float>> face_as_second_result;
        face_as_second_->predict(im_bak0, &face_as_second_result);
        std::vector<float> passive_results;
        std::vector<DetectionLandmarkResult> face_det_result;
        const bool has_box = !face_as_second_result.empty();
        if (!face_det_->predict(im_bak1, &face_det_result)) {
            return false;
        }
        results->resize(face_det_result.size());
        auto align_im_list = utils::align_face_with_five_points(im, face_det_result);
        for (auto& align_image : align_im_list) {
            float passive_result;
            const auto clarity = clarity_estimate(align_image);
            face_as_first_->predict(align_image, &passive_result);
            const float result = has_box ? 0.0f : passive_result;
            if (result > fuse_threshold) {
                if (clarity >= clarity_threshold) {
                    results->push_back(FaceAntiSpoofResult::REAL);
                }
                else {
                    results->push_back(FaceAntiSpoofResult::FUZZY);
                }
            }
            else {
                results->push_back(FaceAntiSpoofResult::SPOOF);
            }
        }
        return true;
    }
}

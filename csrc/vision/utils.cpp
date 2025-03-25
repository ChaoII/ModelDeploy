//
// Created by aichao on 2025/2/20.
//

#include "csrc/vision/utils.h"

#include <execution>

#include "csrc/core/md_log.h"

namespace modeldeploy::vision::utils {
    MDDataType::Type cv_dtype_to_md_dtype(int type) {
        type = type % 8;
        if (type == 0) {
            return MDDataType::Type::UINT8;
        }
        if (type == 1) {
            return MDDataType::Type::INT8;
        }
        if (type == 2) {
            MD_LOG_ERROR("While calling OpenCVDataTypeToMD(), get UINT16 type which is not supported now.");
            return MDDataType::Type::UNKNOWN1;
        }
        if (type == 3) {
            return MDDataType::Type::INT16;
        }
        if (type == 4) {
            return MDDataType::Type::INT32;
        }
        if (type == 5) {
            return MDDataType::Type::FP32;
        }
        if (type == 6) {
            return MDDataType::Type::FP64;
        }
        MD_LOG_ERROR("While calling OpenCVDataTypeToMD(), get type = {}, which is not expected.", type);
        return MDDataType::Type::UNKNOWN1;
    }


    bool mat_to_tensor(cv::Mat& mat, MDTensor* tensor, const bool is_copy) {
        if (is_copy) {
            const int num_bytes = mat.rows * mat.cols * mat.channels() * MDDataType::size(
                cv_dtype_to_md_dtype(mat.type()));
            if (num_bytes != tensor->total_bytes()) {
                MD_LOG_ERROR("While copy Mat to Tensor, requires the memory size be same, "
                             "but now size of Tensor = {}, size of Mat = {}.", tensor->total_bytes(), num_bytes);
                return false;
            }
            memcpy(tensor->mutable_data(), mat.data, num_bytes);
        }
        else {
            tensor->set_external_data({mat.channels(), mat.rows, mat.cols}, cv_dtype_to_md_dtype(mat.type()), mat.data);
        }
        return true;
    }


    bool mats_to_tensor(const std::vector<cv::Mat>& mats, MDTensor* tensor) {
        // Each mat has its own tensor,
        // to get a batched tensor, we need copy these tensors to a batched tensor
        const std::vector<int64_t> shape = {
            static_cast<long long>(mats.size()), mats[0].channels(), mats[0].rows, mats[0].cols
        };
        tensor->resize(shape, cv_dtype_to_md_dtype(mats[0].type()), "batch_tensor");
        for (size_t i = 0; i < mats.size(); ++i) {
            auto* p = static_cast<uint8_t*>(tensor->data());
            const int total_bytes = mats[i].rows * mats[i].cols * mats[i].channels() * MDDataType::size(
                cv_dtype_to_md_dtype(mats[i].type()));
            MDTensor::copy_buffer(p + i * total_bytes, mats[i].data, total_bytes);
        }
        return true;
    }


    void nms(DetectionResult* output, const float iou_threshold,
             std::vector<int>* index) {
        // get sorted score indices
        std::vector<int> sorted_indices;
        if (index != nullptr) {
            std::map<float, int, std::greater<>> score_map;
            for (size_t i = 0; i < output->scores.size(); ++i) {
                score_map.insert(std::pair<float, int>(output->scores[i], i));
            }
            for (auto iter : score_map) {
                sorted_indices.push_back(iter.second);
            }
        }
        sort_detection_result(output);
        std::vector<float> area_of_boxes(output->boxes.size());
        std::vector suppressed(output->boxes.size(), 0);
        for (size_t i = 0; i < output->boxes.size(); ++i) {
            area_of_boxes[i] = (output->boxes[i][2] - output->boxes[i][0]) *
                (output->boxes[i][3] - output->boxes[i][1]);
        }
        for (size_t i = 0; i < output->boxes.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            for (size_t j = i + 1; j < output->boxes.size(); ++j) {
                if (suppressed[j] == 1) {
                    continue;
                }
                float xmin = std::max(output->boxes[i][0], output->boxes[j][0]);
                float ymin = std::max(output->boxes[i][1], output->boxes[j][1]);
                float xmax = std::min(output->boxes[i][2], output->boxes[j][2]);
                float ymax = std::min(output->boxes[i][3], output->boxes[j][3]);
                float overlap_w = std::max(0.0f, xmax - xmin);
                float overlap_h = std::max(0.0f, ymax - ymin);
                float overlap_area = overlap_w * overlap_h;
                float overlap_ratio =
                    overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
                if (overlap_ratio > iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
        DetectionResult backup(*output);
        output->Clear();
        output->Reserve(static_cast<int>(suppressed.size()));
        for (size_t i = 0; i < suppressed.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            output->boxes.emplace_back(backup.boxes[i]);
            output->scores.push_back(backup.scores[i]);
            output->label_ids.push_back(backup.label_ids[i]);
            if (index != nullptr) {
                index->push_back(sorted_indices[i]);
            }
        }
    }


    void nms(FaceDetectionResult* result, float iou_threshold) {
        utils::sort_detection_result(result);

        std::vector<float> area_of_boxes(result->boxes.size());
        std::vector<int> suppressed(result->boxes.size(), 0);
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0]) *
                (result->boxes[i][3] - result->boxes[i][1]);
        }

        for (size_t i = 0; i < result->boxes.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            for (size_t j = i + 1; j < result->boxes.size(); ++j) {
                if (suppressed[j] == 1) {
                    continue;
                }
                float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
                float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
                float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
                float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
                float overlap_w = std::max(0.0f, xmax - xmin);
                float overlap_h = std::max(0.0f, ymax - ymin);
                float overlap_area = overlap_w * overlap_h;
                float overlap_ratio =
                    overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
                if (overlap_ratio > iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
        FaceDetectionResult backup(*result);
        int landmarks_per_face = result->landmarks_per_face;

        result->Clear();
        // don't forget to reset the landmarks_per_face
        // before apply Reserve method.
        result->landmarks_per_face = landmarks_per_face;
        result->Reserve(suppressed.size());
        for (size_t i = 0; i < suppressed.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            result->boxes.emplace_back(backup.boxes[i]);
            result->scores.push_back(backup.scores[i]);
            // landmarks (if have)
            if (result->landmarks_per_face > 0) {
                for (size_t j = 0; j < result->landmarks_per_face; ++j) {
                    result->landmarks.emplace_back(
                        backup.landmarks[i * result->landmarks_per_face + j]);
                }
            }
        }
    }

    cv::Mat center_crop(const cv::Mat& image, const cv::Size& crop_size) {
        // 获取输入图像的尺寸
        int img_height = image.rows;
        int img_width = image.cols;

        // 获取裁剪尺寸
        int crop_height = crop_size.height;
        int crop_width = crop_size.width;

        // 检查裁剪尺寸是否大于输入图像尺寸
        if (crop_height > img_height || crop_width > img_width) {
            std::cerr << "Crop size is larger than the input image size." << std::endl;
            return image; // 或者抛出异常
        }

        // 计算裁剪区域的起始坐标
        int top = (img_height - crop_height) / 2;
        int left = (img_width - crop_width) / 2;

        // 使用子矩阵操作进行裁剪
        cv::Mat cropped_image = image(cv::Rect(left, top, crop_width, crop_height));

        return cropped_image;
    }

    void print_mat_type(const cv::Mat& mat) {
        const int type = mat.type();
        std::string r;
        const uchar depth = type & CV_MAT_DEPTH_MASK;
        const uchar chans = 1 + (type >> CV_CN_SHIFT);
        switch (depth) {
        case CV_8U: r = "8U";
            break;
        case CV_8S: r = "8S";
            break;
        case CV_16U: r = "16U";
            break;
        case CV_16S: r = "16S";
            break;
        case CV_32S: r = "32S";
            break;
        case CV_32F: r = "32F";
            break;
        case CV_64F: r = "64F";
            break;
        default: r = "User";
            break;
        }
        r += "C";
        r += chans + '0';
        std::cout << "Mat type: " << r << std::endl;
    }


    std::vector<float> compute_sqrt(const std::vector<float>& vec) {
        std::vector<float> result(vec.size());
        std::transform(std::execution::par, vec.begin(), vec.end(), result.begin(), [](float x) {
            return std::sqrt(x);
        });
        return result;
    }

    float compute_similarity(const std::vector<float>& feature1, const std::vector<float>& feature2) {
        if (feature1.size() != feature2.size()) {
            MD_LOG_ERROR("The size of feature1 and feature2 should be same.");
            return 0.0f;
        }
        float sum = 0;
        for (int i = 0; i < feature1.size(); ++i) {
            sum += feature1[i] * feature2[i];
        }
        return std::max<float>(sum, 0.0f);
    }

    std::vector<float> l2_normalize(const std::vector<float>& values) {
        size_t num_val = values.size();
        if (num_val == 0) {
            return {};
        }
        std::vector<float> norm;
        float l2_sum_val = 0.f;
        for (size_t i = 0; i < num_val; ++i) {
            l2_sum_val += (values[i] * values[i]);
        }
        float l2_sum_sqrt = std::sqrt(l2_sum_val);
        norm.resize(num_val);
        for (size_t i = 0; i < num_val; ++i) {
            norm[i] = values[i] / l2_sum_sqrt;
        }
        return norm;
    }
}

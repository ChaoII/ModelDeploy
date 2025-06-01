//
// Created by aichao on 2025/2/20.
//


#include <execution>
#include <ranges>

#include "csrc/vision/utils.h"
#include "csrc/core/md_log.h"

namespace modeldeploy::vision::utils
{
    DataType cv_dtype_to_md_dtype(int type) {
        type = type % 8;
        if (type == 0) {
            return DataType::UINT8;
        }
        if (type == 1) {
            return DataType::INT8;
        }
        if (type == 2) {
            MD_LOG_ERROR << "While calling cv_dtype_to_md_dtype(), "
                "get UINT16 type which is not supported now." << std::endl;
            return DataType::UNKNOW;
        }
        if (type == 4) {
            return DataType::INT32;
        }
        if (type == 5) {
            return DataType::FP32;
        }
        if (type == 6) {
            return DataType::FP64;
        }

        MD_LOG_ERROR << "While calling cv_dtype_to_md_dtype(), get type = "
            << type << ", which is not expected." << std::endl;
        return DataType::UNKNOW;
    }


    bool mat_to_tensor(cv::Mat& mat, Tensor* tensor, const bool is_copy) {
        const auto dtype = cv_dtype_to_md_dtype(mat.type());
        if (is_copy) {
            const size_t num_bytes = mat.rows * mat.cols * mat.channels() * Tensor::get_element_size(dtype);
            tensor->allocate({mat.channels(), mat.rows, mat.cols}, dtype);
            if (num_bytes != tensor->byte_size()) {
                MD_LOG_ERROR << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = " << tensor->byte_size()
                    << ", size of Mat = " << num_bytes << "." << std::endl;
                return false;
            }
            memcpy(tensor->data(), mat.data, num_bytes);
        }
        else {
            // OpenCV Mat 的内存管理由 Mat 自己处理，这里不需要额外操作
            // 注意tensor共享外部内存，所以需要从外部内存中创建tensor，内存由Mat提供，所以deleter可以不给，不需要进行手动释放
            // 确保mat在tensor生命周期结束前有效
            tensor->from_external_memory(mat.data, {mat.channels(), mat.rows, mat.cols}, dtype);
        }
        return true;
    }


    bool mats_to_tensor(const std::vector<cv::Mat>& mats, Tensor* tensor) {
        // Each mat has its own tensor,
        // to get a batched tensor, we need copy these tensors to a batched tensor
        const std::vector<int64_t> shape = {
            static_cast<long long>(mats.size()), mats[0].channels(), mats[0].rows, mats[0].cols
        };
        const auto dtype = cv_dtype_to_md_dtype(mats[0].type());
        const size_t total_bytes = mats[0].rows * mats[0].cols * mats[0].channels() * Tensor::get_element_size(dtype);
        tensor->allocate(shape, dtype);
        for (size_t i = 0; i < mats.size(); ++i) {
            auto* p = static_cast<uint8_t*>(tensor->data());
            std::memcpy(p + i * total_bytes, mats[i].data, total_bytes);
        }
        return true;
    }


    void nms(DetectionResult* output, const float iou_threshold, std::vector<int>* index) {
        // get sorted score indices
        std::vector<int> sorted_indices;
        if (index != nullptr) {
            std::map<float, int, std::greater<>> score_map;
            for (size_t i = 0; i < output->scores.size(); ++i) {
                score_map.insert({output->scores[i], static_cast<int>(i)});
            }
            for (auto val : score_map | std::views::values) {
                sorted_indices.push_back(val);
            }
        }
        sort_detection_result(output);
        std::vector<float> area_of_boxes(output->boxes.size());
        std::vector suppressed(output->boxes.size(), 0);
        for (size_t i = 0; i < output->boxes.size(); ++i) {
            area_of_boxes[i] = output->boxes[i].width * output->boxes[i].height;
        }
        for (size_t i = 0; i < output->boxes.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            for (size_t j = i + 1; j < output->boxes.size(); ++j) {
                if (suppressed[j] == 1) continue;
                // 手动计算
                // const float xmin = std::max(output->boxes[i].x, output->boxes[j].x);
                // const float ymin = std::max(output->boxes[i].y, output->boxes[j].y);
                // const float xmax = std::min(output->boxes[i].x + output->boxes[i].width,
                //                             output->boxes[j].x + output->boxes[j].width);
                // const float ymax = std::min(output->boxes[i].y + output->boxes[i].height,
                //                             output->boxes[j].y + output->boxes[j].height);
                //
                // const float overlap_w = std::max(0.0f, xmax - xmin);
                // const float overlap_h = std::max(0.0f, ymax - ymin);
                // const float overlap_area = overlap_w * overlap_h;
                // const float overlap_ratio =
                //     overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
                // if (overlap_ratio > iou_threshold) {
                //     suppressed[j] = 1;
                // }

                // 使用opencv的api
                const cv::Rect2f& box_i = output->boxes[i];
                const cv::Rect2f& box_j = output->boxes[j];

                cv::Rect2f intersection = box_i & box_j; // 取交集
                const float inter_area = intersection.area();
                const float union_area = area_of_boxes[i] + area_of_boxes[j] - inter_area;

                const float iou = inter_area / union_area;
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        DetectionResult backup(*output);
        output->clear();
        output->reserve(static_cast<int>(suppressed.size()));
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


    void nms(DetectionLandmarkResult* result, float iou_threshold) {
        utils::sort_detection_result(result);
        std::vector<float> area_of_boxes(result->boxes.size());
        std::vector suppressed(result->boxes.size(), false);
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            area_of_boxes[i] = result->boxes[i].width * result->boxes[i].height;
        }
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            for (size_t j = i + 1; j < result->boxes.size(); ++j) {
                if (suppressed[j]) continue;
                // 手动计算iou
                // const float xmin = std::max(result->boxes[i].x, result->boxes[j].x);
                // const float ymin = std::max(result->boxes[i].y, result->boxes[j].y);
                // const float xmax = std::min(result->boxes[i].x + result->boxes[i].width,
                //                             result->boxes[j].x + result->boxes[j].width);
                // const float ymax = std::min(result->boxes[i].y + result->boxes[i].height,
                //                             result->boxes[j].y + result->boxes[j].height);
                // const float overlap_w = std::max(0.0f, xmax - xmin);
                // const float overlap_h = std::max(0.0f, ymax - ymin);
                // const float overlap_area = overlap_w * overlap_h;
                // const float overlap_ratio =
                //     overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
                // if (overlap_ratio > iou_threshold) {
                //     suppressed[j] = 1;
                // }

                // 使用opencv的api
                const cv::Rect2f& box_i = result->boxes[i];
                const cv::Rect2f& box_j = result->boxes[j];

                cv::Rect2f intersection = box_i & box_j; // 取交集
                const float inter_area = intersection.area();
                const float union_area = area_of_boxes[i] + area_of_boxes[j] - inter_area;

                const float iou = inter_area / union_area;
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        DetectionLandmarkResult backup(*result);
        const int landmarks_per_face = result->landmarks_per_instance;
        result->clear();
        // don't forget to reset the landmarks_per_face
        // before apply Reserve method.
        result->landmarks_per_instance = landmarks_per_face;
        result->reserve(suppressed.size());
        for (size_t i = 0; i < suppressed.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            result->boxes.emplace_back(backup.boxes[i]);
            result->scores.push_back(backup.scores[i]);
            result->label_ids.push_back(backup.label_ids[i]);
            // landmarks (if have)
            if (result->landmarks_per_instance > 0) {
                for (size_t j = 0; j < result->landmarks_per_instance; ++j) {
                    result->landmarks.emplace_back(
                        backup.landmarks[i * result->landmarks_per_instance + j]);
                }
            }
        }
    }

    cv::Mat center_crop(const cv::Mat& image, const cv::Size& crop_size) {
        // 获取输入图像的尺寸
        const int img_height = image.rows;
        const int img_width = image.cols;
        // 获取裁剪尺寸
        const int crop_height = crop_size.height;
        const int crop_width = crop_size.width;

        // 检查裁剪尺寸是否大于输入图像尺寸
        if (crop_height > img_height || crop_width > img_width) {
            MD_LOG_ERROR << "Crop size is larger than the input image size." << std::endl;
            return image; // 或者抛出异常
        }
        // 计算裁剪区域的起始坐标
        const int top = (img_height - crop_height) / 2;
        const int left = (img_width - crop_width) / 2;
        // 使用子矩阵操作进行裁剪, 裁剪后cv::Mat 内存不连续，需要执行clone()操作
        const cv::Mat cropped_image = image(cv::Rect(left, top, crop_width, crop_height));
        return cropped_image.clone();
    }

    void print_mat_type(const cv::Mat& mat) {
        const int type = mat.type();
        std::string r;
        const uchar depth = type & CV_MAT_DEPTH_MASK;
        const uchar chans = 1 + (type >> CV_CN_SHIFT);
        switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
        }
        r += "C";
        r += chans + '0';
        std::cout << "Mat type: " << r << std::endl;
    }

    std::vector<float> compute_sqrt(const std::vector<float>& vec) {
        std::vector<float> result(vec.size());
        std::transform(std::execution::par, vec.begin(), vec.end(), result.begin(),
                       [](const float x) {
                           return std::sqrt(x);
                       });
        return result;
    }

    float compute_similarity(const std::vector<float>& feature1, const std::vector<float>& feature2) {
        if (feature1.size() != feature2.size()) {
            MD_LOG_ERROR << "The size of feature1 and feature2 should be same." << std::endl;
            return 0.0f;
        }
        float sum = 0;
        for (int i = 0; i < feature1.size(); ++i) {
            sum += feature1[i] * feature2[i];
        }
        return std::max<float>(sum, 0.0f);
    }

    std::vector<float> l2_normalize(const std::vector<float>& values) {
        const size_t num_val = values.size();
        if (num_val == 0) {
            return {};
        }
        std::vector<float> norm;
        float l2_sum_val = 0.f;
        for (size_t i = 0; i < num_val; ++i) {
            l2_sum_val += values[i] * values[i];
        }
        const float l2_sum_sqrt = std::sqrt(l2_sum_val);
        norm.resize(num_val);
        for (size_t i = 0; i < num_val; ++i) {
            norm[i] = values[i] / l2_sum_sqrt;
        }
        return norm;
    }

    std::array<float, 8> xcycwha_to_x1y1x2y2x3y3x4y4(const float xc, const float yc, const float w, const float h,
                                                     const float angle_rad) {
        // 半宽高
        const float cos_a = std::cos(angle_rad);
        const float sin_a = std::sin(angle_rad);
        const float dx = w / 2.0f;
        const float dy = h / 2.0f;

        // 四个顶点相对于中心点的偏移（顺时针）
        const float x0 = -dx, y0 = -dy;
        const float x1 = dx, y1 = -dy;
        const float x2 = dx, y2 = dy;
        const float x3 = -dx, y3 = dy;

        // 旋转 + 平移到中心点
        const std::array points = {
            xc + cos_a * x0 - sin_a * y0, yc + sin_a * x0 + cos_a * y0, // x1, y1
            xc + cos_a * x1 - sin_a * y1, yc + sin_a * x1 + cos_a * y1, // x2, y2
            xc + cos_a * x2 - sin_a * y2, yc + sin_a * x2 + cos_a * y2, // x3, y3
            xc + cos_a * x3 - sin_a * y3, yc + sin_a * x3 + cos_a * y3 // x4, y4
        };
        return points;
    }

    std::array<float, 5> x1y1x2y2x3y3x4y4_to_xcycwha(const std::array<float, 8>& pts) {
        // 提取四个点
        const float x1 = pts[0], y1 = pts[1];
        const float x2 = pts[2], y2 = pts[3];
        const float x3 = pts[4], y3 = pts[5];
        const float x4 = pts[6], y4 = pts[7];
        // 中心点 (平均四点)
        const float xc = (x1 + x2 + x3 + x4) / 4.0f;
        const float yc = (y1 + y2 + y3 + y4) / 4.0f;
        // 宽 w = p1 -> p2 的距离
        const float dx_w = x2 - x1;
        const float dy_w = y2 - y1;
        const float w = std::sqrt(dx_w * dx_w + dy_w * dy_w);
        // 高 h = p2 -> p3 的距离
        const float dx_h = x3 - x2;
        const float dy_h = y3 - y2;
        const float h = std::sqrt(dx_h * dx_h + dy_h * dy_h);
        // 角度（p1 -> p2）方向，atan2(y, x)
        const float angle = std::atan2(dy_w, dx_w); // 弧度制，逆时针为正
        return {xc, yc, w, h, angle};
    }
}

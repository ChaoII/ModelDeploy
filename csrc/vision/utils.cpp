//
// Created by aichao on 2025/2/20.
//


#include <execution>
#include <ranges>

#include "csrc/vision/utils.h"
#include "csrc/core/md_log.h"

namespace modeldeploy::vision::utils {
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


    float rect_iou(const cv::Rect2f& rect1, const cv::Rect2f& rect2) {
        // 手动计算
        // const float xmin = std::max(rect1.x, rect2.x);
        // const float ymin = std::max(rect1.y, rect2.y);
        // const float xmax = std::min(rect1.x + rect1.width,
        //                             rect2.x + rect2.width);
        // const float ymax = std::min(rect1.y + rect1.height,
        //                             rect2.y + rect2.height);
        //
        // const float overlap_w = std::max(0.0f, xmax - xmin);
        // const float overlap_h = std::max(0.0f, ymax - ymin);
        // const float overlap_area = overlap_w * overlap_h;
        // const float area1 = rect1.width * rect1.height;
        // const float area2 = rect2.width * rect2.height;
        // const float iou = overlap_area / (area1 + area2 - overlap_area);


        // 使用opencv的api
        cv::Rect2f intersection = rect1 & rect2; // 取交集
        const float inter_area = intersection.area();
        const float union_area = rect1.area() + rect2.area() - inter_area;
        const float iou = inter_area / union_area;
        return iou;
    }


    void nms(DetectionResult* result, const float iou_threshold, std::vector<int>* index) {
        // get sorted score indices
        const size_t N = result->boxes.size();
        // Step 1: 根据分数排序得到索引
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::ranges::sort(sorted_indices, [&](const int a, const int b) {
            return result->scores[a] > result->scores[b]; // 分数高的排前面
        });

        // Step 2: NMS 主逻辑
        std::vector suppressed(N, false);
        std::vector<int> keep_indices;
        for (size_t m = 0; m < N; ++m) {
            int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep_indices.push_back(i); // 保留当前框
            const auto& box_i = result->boxes[i];
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& box_j = result->boxes[j];
                const float iou = rect_iou(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        // Step 3: 根据 keep_indices 重建结果
        const DetectionResult backup(*result);
        result->clear();
        result->reserve(static_cast<int>(keep_indices.size()));
        for (int idx : keep_indices) {
            result->boxes.push_back(backup.boxes[idx]);
            result->scores.push_back(backup.scores[idx]);
            result->label_ids.push_back(backup.label_ids[idx]);
            if (index) {
                index->push_back(idx); // 保留原始下标
            }
        }
    }


    void nms(std::vector<PoseResult>* result, const float iou_threshold) {
        const size_t N = result->size();
        // Step 1: 根据分数排序得到索引
        std::ranges::sort(*result, [&](const PoseResult& a, const PoseResult& b) {
            return a.score > b.score; // 分数高的排前面
        });

        std::vector<size_t> index_;
        // Step 2: NMS 主逻辑
        std::vector<bool> suppressed(N);
        for (size_t m = 0; m < N; ++m) {
            if (suppressed[m]) continue;
            index_.push_back(m);
            const auto& box_i = result->at(m).box;
            for (size_t n = m + 1; n < N; ++n) {
                if (suppressed[n]) continue;
                const auto& box_j = result->at(n).box;
                const float iou = rect_iou(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[n] = true;
                }
            }
        }

        // Step 3: 根据 keep_indices 重建结果
        const std::vector<PoseResult> backup = *result;
        result->clear();
        for (const auto idx : index_) {
            result->push_back(backup[idx]);
        }
    }


    void nms(DetectionLandmarkResult* result, float iou_threshold) {
        const size_t N = result->boxes.size();
        // Step 1: 根据分数排序得到索引
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::ranges::sort(sorted_indices, [&](const int a, const int b) {
            return result->scores[a] > result->scores[b]; // 分数高的排前面
        });

        // Step 2: NMS 主逻辑
        std::vector suppressed(N, false);
        std::vector<int> keep_indices;

        for (size_t m = 0; m < N; ++m) {
            int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep_indices.push_back(i); // 保留当前框
            const auto& box_i = result->boxes[i];
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& box_j = result->boxes[j];
                const float iou = rect_iou(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        DetectionLandmarkResult backup(*result);
        result->clear();
        // don't forget to reset the landmarks_per_face
        // before apply Reserve method.
        result->landmarks_per_instance = result->landmarks_per_instance;
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

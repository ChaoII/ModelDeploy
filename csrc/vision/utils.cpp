//
// Created by aichao on 2025/2/20.
//


#include "vision/utils.h"
#include "core/md_log.h"
#include <numeric>

namespace modeldeploy::vision::utils {
    LetterBoxRecord cal_letter_box_param(const std::vector<int>& src_size, const std::vector<int>& dst_size) {
        const float src_w = static_cast<float>(src_size[0]);
        const float src_h = static_cast<float>(src_size[1]);
        const float dst_w = static_cast<float>(dst_size[0]);
        const float dst_h = static_cast<float>(dst_size[1]);
        const float scale = std::min(dst_h / src_h, dst_w / src_w);
        const float resize_w = src_w * scale;
        const float resize_h = src_h * scale;
        const float pad_w = (dst_w - resize_w) * 0.5f;
        const float pad_h = (dst_h - resize_h) * 0.5f;
        return {src_w, src_h, dst_w, dst_h, pad_w, pad_h, scale};
    }


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
            return DataType::UNKNOWN;
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
        return DataType::UNKNOWN;
    }

    DataType md_image_dtype_to_md_dtype(MdImageType type) {
        if (type == MdImageType::GRAY_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_BGR_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_RGB_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PLA_BGR_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PLA_RGB_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PLA_BGRA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PLA_RGBA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PLA_BGR_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PLA_RGB_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PLA_BGRA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PLA_RGBA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_BGRA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_RGBA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_BGRA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_RGBA_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_BGR565_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::PKG_RGB565_U8) {
            return DataType::UINT8;
        }
        if (type == MdImageType::GRAY_S32) {
            return DataType::INT32;
        }
        if (type == MdImageType::GRAY_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_BGR_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_RGB_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_BGR_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_RGB_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_BGRA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_RGBA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_BGRA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::PKG_RGBA_F32) {
            return DataType::FP32;
        }
        if (type == MdImageType::GRAY_F64) {
            return DataType::FP64;
        }
        MD_LOG_ERROR << "While calling md_image_dtype_to_md_dtype(), get unexpected type:" <<
            md_image_type_to_string(type);
        return DataType::UNKNOWN;
    }

    cv::Point2f point2f_to_cv_type(const Point2f point2f) {
        return cv::Point2f{point2f.x, point2f.y};
    }

    cv::Point3f point3f_to_cv_type(Point3f point3f) {
        return {point3f.x, point3f.y, point3f.z};
    }

    cv::Rect2f rect2f_to_cv_type(Rect2f rect2f) {
        return {rect2f.x, rect2f.y, rect2f.width, rect2f.height};
    }

    // 内联标量 IoU（与 cv::Rect2f 的 operator& / area() 数值一致，但无函数调用开销）
    float iou_rects(const Rect2f& r1, const Rect2f& r2) {
        const float xmin = r1.x > r2.x ? r1.x : r2.x;
        const float ymin = r1.y > r2.y ? r1.y : r2.y;
        const float xmax1 = r1.x + r1.width;
        const float ymax1 = r1.y + r1.height;
        const float xmax2 = r2.x + r2.width;
        const float ymax2 = r2.y + r2.height;
        const float xmax = xmax1 < xmax2 ? xmax1 : xmax2;
        const float ymax = ymax1 < ymax2 ? ymax1 : ymax2;
        const float overlap_w = xmax - xmin > 0 ? xmax - xmin : 0.0f;
        const float overlap_h = ymax - ymin > 0 ? ymax - ymin : 0.0f;
        const float inter = overlap_w * overlap_h;
        const float area1 = r1.width * r1.height;
        const float area2 = r2.width * r2.height;
        const float uni = area1 + area2 - inter;
        return uni > 0 ? inter / uni : 0.0f;
    }

    cv::RotatedRect rotated_rect_to_cv_type(RotatedRect rotated_rect) {
        return {
            cv::Point2f(rotated_rect.xc, rotated_rect.yc),
            cv::Point2f(rotated_rect.width, rotated_rect.height),
            rotated_rect.angle
        };
    }

    bool image_data_to_tensor(const ImageData* image_data, Tensor* tensor) {
        cv::Mat mat;
        image_data->asMat(&mat);
        return mat_to_tensor(mat, tensor);
    }

    // cv::Mat image_data_to_mat(ImageData& image) {
    //     return {image.height(), image.width(), image.type(), image.data()};
    // }

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

    bool image_data_to_tensor(ImageData& image, Tensor* tensor, const bool is_copy) {
        const auto dtype = md_image_dtype_to_md_dtype(image.type());
        if (is_copy) {
            const size_t num_bytes = image.height() * image.width() * image.channels() *
                Tensor::get_element_size(dtype);
            tensor->allocate({image.channels(), image.height(), image.width()}, dtype);
            if (num_bytes != tensor->byte_size()) {
                MD_LOG_ERROR << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = " << tensor->byte_size()
                    << ", size of Mat = " << num_bytes << "." << std::endl;
                return false;
            }
            memcpy(tensor->data(), image.data(), num_bytes);
        }
        else {
            // OpenCV Mat 的内存管理由 Mat 自己处理，这里不需要额外操作
            // 注意tensor共享外部内存，所以需要从外部内存中创建tensor，内存由Mat提供，所以deleter可以不给，不需要进行手动释放
            // 确保mat在tensor生命周期结束前有效
            tensor->from_external_memory(image.data(), {image.channels(), image.height(), image.width()}, dtype);
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

    void sorted_det_land_mark_results(std::vector<KeyPointsResult>& results) {
        std::sort(results.begin(), results.end(), [](const KeyPointsResult& a,
                                                     const KeyPointsResult& b) {
            return a.box.width * a.box.height > b.box.width * b.box.height;
        });
    }

    void sorted_det_results(std::vector<DetectionResult>& results) {
        std::sort(results.begin(), results.end(), [](const DetectionResult& a,
                                                      const DetectionResult& b) {
            return a.box.width * a.box.height > b.box.width * b.box.height;
        });
    }

    void nms(std::vector<DetectionResult>* result, const float iou_threshold, std::vector<int>* index) {
        const size_t N = result->size();
        if (N <= 1) return;
        // 复用临时缓冲，避免每帧堆分配
        static thread_local std::vector<int> sorted_indices;
        static thread_local std::vector<uint8_t> suppressed;
        static thread_local std::vector<int> keep_indices;
        static thread_local std::vector<DetectionResult> new_result;
        sorted_indices.resize(N);
        suppressed.assign(N, 0);
        keep_indices.clear();
        new_result.clear();
        new_result.reserve(N);
        // Step 1: 根据分数排序得到索引
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const int a, const int b) {
            return (*result)[a].score > (*result)[b].score; // 分数高的排前面
        });

        // Step 2: NMS 主逻辑（沿用原 rect_iou 保证数值一致）
        for (size_t m = 0; m < N; ++m) {
            const int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep_indices.push_back(i); // 保留当前框
            const auto& box_i = (*result)[i].box;
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& box_j = (*result)[j].box;
                const float iou = iou_rects(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
        // Step 3: 根据 keep_indices 重建结果
        for (const auto idx : keep_indices) {
            new_result.push_back(std::move((*result)[idx])); // 移动语义
            if (index) {
                index->push_back(idx);
            }
        }
        result->swap(new_result);
    }

    void nms(std::vector<InstanceSegResult>* result, const float iou_threshold, std::vector<int>* index) {
        // get sorted score indices
        const size_t N = result->size();
        // Step 1: 根据分数排序得到索引
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // 初始化索引 [0, 1, ..., N-1]
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const int a, const int b) {
            return (*result)[a].score > (*result)[b].score; // 分数高的排前面
        });

        // Step 2: NMS 主逻辑
        std::vector suppressed(N, false);
        std::vector<int> keep_indices;

        for (size_t m = 0; m < N; ++m) {
            int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep_indices.push_back(i); // 保留当前框
            const auto& box_i = (*result)[i].box;
            for (size_t n = m + 1; n < N; ++n) {
                int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& box_j = (*result)[j].box;
                const float iou = iou_rects(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        // Step 3: 根据 keep_indices 重建结果
        std::vector<InstanceSegResult> new_result;
        new_result.reserve(keep_indices.size());
        for (const auto idx : keep_indices) {
            new_result.push_back(std::move((*result)[idx])); // 移动语义
            if (index) {
                index->push_back(idx);
            }
        }
        result->swap(new_result);
    }


    void nms(std::vector<KeyPointsResult>* result, const float iou_threshold) {
        const size_t N = result->size();
        // Step 1: 根据分数排序得到索引
        std::sort(result->begin(), result->end(), [&](const KeyPointsResult& a, const KeyPointsResult& b) {
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
                if (box_i.width * box_i.height <= 0 || box_j.width * box_j.height <= 0) continue;
                const float iou = iou_rects(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[n] = true;
                }
            }
        }
        // Step 3: 根据 keep_indices 重建结果
        std::vector<KeyPointsResult> new_result;
        new_result.reserve(index_.size());
        for (const auto idx : index_) {
            new_result.push_back(std::move((*result)[idx])); // 移动语义
        }
        result->swap(new_result);
    }


    ImageData center_crop(const ImageData& image, const cv::Size& crop_size) {
        // 获取输入图像的尺寸
        const int img_height = image.height();
        const int img_width = image.width();
        // 获取裁剪尺寸
        const int crop_height = crop_size.height;
        const int crop_width = crop_size.width;

        // 检查裁剪尺寸是否大于输入图像尺寸
        if (crop_height > img_height || crop_width > img_width) {
            MD_LOG_ERROR << "Crop size is larger than the input image size." << std::endl;
            return image; // 或者抛出异常
        }
        cv::Mat cv_image;
        image.asMat(&cv_image);
        // 计算裁剪区域的起始坐标
        const int top = (img_height - crop_height) / 2;
        const int left = (img_width - crop_width) / 2;
        // 使用子矩阵操作进行裁剪, 裁剪后cv::Mat 内存不连续，需要执行clone()操作
        const cv::Mat cropped_image = cv_image(cv::Rect(left, top, crop_width, crop_height)).clone();
        return ImageData(std::move(cropped_image));
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
        // 普通串行 transform：sqrt 是逐元素内存带宽受限操作，std::execution::par (GCC PSTL)
        // 依赖 oneTBB，且并行几乎无收益（与内存带宽瓶颈同理）。串行避免引入 TBB 链接依赖。
        std::transform(vec.begin(), vec.end(), result.begin(),
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

//
// Created by aichao on 2025/2/20.
//
#include "utils.h"

namespace modeldeploy::vision {
    MDDataType cv_dtype_to_md_dtype(int type) {
        type = type % 8;
        if (type == 0) {
            return MDDataType::UINT8;
        }
        else if (type == 1) {
            return MDDataType::INT8;
        }
        else if (type == 2) {
            std::cerr << "While calling OpenCVDataTypeToFD(), get UINT16 type which is not "
                "supported now." << std::endl;
            return MDDataType::UNKNOWN1;
        }
        else if (type == 3) {
            return MDDataType::INT16;
        }
        else if (type == 4) {
            return MDDataType::INT32;
        }
        else if (type == 5) {
            return MDDataType::FP32;
        }
        else if (type == 6) {
            return MDDataType::FP64;
        }
        else {
            std::cerr <<
                "While calling OpenCVDataTypeToFD(), get type = " << type << ", which is not "
                "expected." << std::endl;
            return MDDataType::UNKNOWN1;
        }
    }


    bool mat_to_tensor(cv::Mat& mat, MDTensor* tensor, bool is_copy) {
        if (is_copy) {
            int total_bytes = mat.rows * mat.cols * mat.channels() * md_dtype_size(cv_dtype_to_md_dtype(mat.type()));
            if (total_bytes != tensor->total_bytes()) {
                std::cerr << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = "
                    << tensor->total_bytes() << ", size of Mat = " << total_bytes << "."
                    << std::endl << std::endl;
                return false;
            }
            memcpy(tensor->mutable_data(), mat.data, total_bytes);
        }
        else {
            tensor->set_external_data({mat.channels(), mat.rows, mat.cols}, cv_dtype_to_md_dtype(mat.type()), mat.data);
        }
        return true;
    }


    bool mats_to_tensor(std::vector<cv::Mat> mats, MDTensor* tensor) {
        // Each mat has its own tensor,
        // to get a batched tensor, we need copy these tensors to a batched tensor
        std::vector<int64_t> shape = {
            static_cast<long long>(mats.size()), mats[0].channels(), mats[0].rows, mats[0].cols
        };
        tensor->resize(shape, cv_dtype_to_md_dtype(mats[0].type()), "batch_tensor");
        for (size_t i = 0; i < mats.size(); ++i) {
            auto* p = reinterpret_cast<uint8_t*>(tensor->data());
            int total_bytes = mats[i].rows * mats[i].cols * mats[i].channels() * md_dtype_size(
                cv_dtype_to_md_dtype(mats[i].type()));
            MDTensor::copy_buffer(p + i * total_bytes, mats[i].data, total_bytes);
        }
    }


    void nms(DetectionResult* result, float iou_threshold,
             std::vector<int>* index) {
        // get sorted score indices
        std::vector<int> sorted_indices;
        if (index != nullptr) {
            std::map<float, int, std::greater<float>> score_map;
            for (size_t i = 0; i < result->scores.size(); ++i) {
                score_map.insert(std::pair<float, int>(result->scores[i], i));
            }
            for (auto iter : score_map) {
                sorted_indices.push_back(iter.second);
            }
        }
        //  utils::SortDetectionResult(result);

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
        DetectionResult backup(*result);
        result->Clear();
        result->Reserve(suppressed.size());
        for (size_t i = 0; i < suppressed.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            result->boxes.emplace_back(backup.boxes[i]);
            result->scores.push_back(backup.scores[i]);
            result->label_ids.push_back(backup.label_ids[i]);
            if (index != nullptr) {
                index->push_back(sorted_indices[i]);
            }
        }
    }


    void print_mat_type(const cv::Mat& mat) {
        int type = mat.type();
        std::string r;
        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);
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
        r += (chans + '0');
        std::cout << "Mat type: " << r << std::endl;
    }
}

//
// Created by aichao on 2025/2/21.
//

#include <opencv2/opencv.hpp>
#include "vision/ocr/det_postprocessor.h"
#include "vision/ocr/utils/ocr_utils.h"
#include "vision/ocr/utils/ocr_postprocess_op.h"


namespace modeldeploy::vision::ocr {
    bool DBDetectorPostprocessor::single_batch_postprocessor(
        const float* prob, const int h, const int w,
        const std::array<int, 4>& det_img_info,
        std::vector<std::array<int, 8>>* boxes_result) const {
        const cv::Mat prob_map(h, w, CV_32F, const_cast<float*>(prob));
        thread_local cv::Mat bin_map;
        bin_map.create(h, w, CV_8UC1);
        // threshold
        for (int i = 0; i < w * h; ++i) {
            bin_map.data[i] = prob[i] > det_db_thresh_ ? 255 : 0;
        }
        if (use_dilation_) {
            const cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::dilate(bin_map, bin_map, dila_ele);
        }
        auto boxes = PostProcessor::boxes_from_bitmap(
            prob_map, bin_map, static_cast<float>(det_db_box_thresh_),
            static_cast<float>(det_db_unclip_ratio_), det_db_score_mode_);
        boxes = PostProcessor::filter_tag_det_res(boxes, det_img_info);
        // boxes to boxes_result
        for (const auto& box : boxes) {
            std::array<int, 8> new_box{}; //{x0, y0, x1, y1, x2, y2, x3, y3}
            for (int i = 0; i < 4; ++i) {
                new_box[i * 2 + 0] = box[i][0];
                new_box[i * 2 + 1] = box[i][1];
            }
            boxes_result->emplace_back(new_box);
        }
        return true;
    }

    bool DBDetectorPostprocessor::run(
        const std::vector<Tensor>& tensors,
        std::vector<std::vector<std::array<int, 8>>>* results,
        const std::vector<std::array<int, 4>>& batch_img_info) const {
        // DBDetector have only 1 output tensor.
        const Tensor& tensor = tensors[0];
        // For DBDetector, the output tensor shape = [batch, 1, ?, ?]
        const size_t batch = tensor.shape()[0];
        const size_t dim1 = tensor.shape()[1];
        const size_t dim2 = tensor.shape()[2]; //h
        const size_t dim3 = tensor.shape()[3]; //w
        const size_t db_out_length = dim1 * dim2 * dim3;
        auto tensor_data = static_cast<const float*>(tensor.data());
        results->resize(batch);
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            tensor_data = tensor_data + i_batch * db_out_length;
            single_batch_postprocessor(tensor_data, dim2, dim3,
                                       batch_img_info[i_batch],
                                       &results->at(i_batch));
        }
        return true;
    }
}

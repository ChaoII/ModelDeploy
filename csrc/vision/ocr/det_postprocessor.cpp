//
// Created by aichao on 2025/2/21.
//

#include <opencv2/opencv.hpp>
#include "csrc/vision/ocr/det_postprocessor.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    bool DBDetectorPostprocessor::single_batch_postprocessor(
        const float* out_data, int n2, int n3,
        const std::array<int, 4>& det_img_info,
        std::vector<std::array<int, 8>>* boxes_result) {
        int n = n2 * n3;
        // prepare bitmap
        std::vector<float> pred(n, 0.0);
        std::vector<unsigned char> cbuf(n, ' ');
        for (int i = 0; i < n; i++) {
            pred[i] = static_cast<float>(out_data[i]);
            cbuf[i] = static_cast<unsigned char>(out_data[i] * 255);
        }
        cv::Mat cbuf_map(n2, n3, CV_8UC1, cbuf.data());
        cv::Mat pred_map(n2, n3, CV_32F, pred.data());
        const double threshold = det_db_thresh_ * 255;
        double max_value = 255;
        cv::Mat bit_map;
        cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
        if (use_dilation_) {
            cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::dilate(bit_map, bit_map, dila_ele);
        }
        std::vector<std::vector<std::vector<int>>> boxes;
        boxes = util_post_processor_.boxes_from_bitmap(
            pred_map, bit_map, static_cast<float>(det_db_box_thresh_),
            static_cast<float>(det_db_unclip_ratio_), det_db_score_mode_);
        boxes = util_post_processor_.filter_tag_det_res(boxes, det_img_info);
        // boxes to boxes_result
        for (auto& boxe : boxes) {
            std::array<int, 8> new_box{};
            int k = 0;
            for (auto& vec : boxe) {
                for (auto& e : vec) {
                    new_box[k++] = e;
                }
            }
            boxes_result->emplace_back(new_box);
        }
        return true;
    }

    bool DBDetectorPostprocessor::apply(
        const std::vector<MDTensor>& tensors,
        std::vector<std::vector<std::array<int, 8>>>* results,
        const std::vector<std::array<int, 4>>& batch_det_img_info) {
        // DBDetector have only 1 output tensor.

        const MDTensor& tensor = tensors[0];
        // For DBDetector, the output tensor shape = [batch, 1, ?, ?]
        const size_t batch = tensor.shape[0];
        const size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                                         std::multiplies());
        auto tensor_data = static_cast<const float*>(tensor.data());
        results->resize(batch);
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            single_batch_postprocessor(tensor_data, static_cast<int>(tensor.shape[2]),
                                       static_cast<int>(tensor.shape[3]),
                                       batch_det_img_info[i_batch],
                                       &results->at(i_batch));

            tensor_data = tensor_data + length;
        }
        return true;
    }
}

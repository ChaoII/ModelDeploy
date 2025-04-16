//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision/ocr/structurev2_layout_postprocessor.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"
#include "csrc/vision/utils.h"
#include "csrc/core/md_log.h"
#include <numeric>


namespace modeldeploy::vision::ocr {
    bool StructureV2LayoutPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<DetectionResult>* results,
        const std::vector<std::array<int, 4>>& batch_layout_img_info) {
        // A StructureV2Layout has 8 output tensors on which it then runs
        // a GFL regression (namely, DisPred2Box), reference:
        // PaddleOCR/blob/release/2.6/deploy/cpp_infer/src/postprocess_op.cpp#L511
        int tensor_size = tensors.size();
        if (tensor_size != 8) {
            MD_LOG_ERROR << "StructureV2Layout should has 8 output tensors,"
                "but got " << tensor_size << " now!" << std::endl;
        }
        if ((tensor_size / 2) != fpn_stride_.size()) {
            MD_LOG_ERROR << "found (tensor_size / 2) != fpn_stride_.size() !" << std::endl;
        }
        // TODO: may need to reorder the tensors according to
        // fpn_stride_ and the shape of output tensors.
        size_t batch = tensors[0].shape()[0]; // [batch, ...]

        results->resize(batch);
        set_reg_max(tensors[fpn_stride_.size()].shape()[2] / 4);
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            std::vector<Tensor> single_batch_tensors(8);
            set_single_batch_external_data(tensors, single_batch_tensors, batch_idx);
            single_batch_postprocessor(single_batch_tensors,
                                       batch_layout_img_info[batch_idx],
                                       &results->at(batch_idx));
        }
        return true;
    }

    void StructureV2LayoutPostprocessor::set_single_batch_external_data(
        const std::vector<Tensor>& tensors,
        std::vector<Tensor>& single_batch_tensors, size_t batch_idx) {
        single_batch_tensors.resize(tensors.size());
        for (int j = 0; j < tensors.size(); ++j) {
            auto j_shape = tensors[j].shape();
            j_shape[0] = 1; // process b=1 per loop
            size_t j_step =
                std::accumulate(j_shape.begin(), j_shape.end(), 1, std::multiplies());
            const float* j_data_ptr = reinterpret_cast<const float*>(tensors[j].data());
            const float* j_start_ptr = j_data_ptr + j_step * batch_idx;
            // todo error
            // single_batch_tensors[j].set_external_data(
            //     j_shape, tensors[j].dtype(),
            //     const_cast<void*>(reinterpret_cast<const void*>(j_start_ptr)));
        }
    }

    bool StructureV2LayoutPostprocessor::single_batch_postprocessor(
        const std::vector<Tensor>& single_batch_tensors,
        const std::array<int, 4>& layout_img_info, DetectionResult* result) {
        if (single_batch_tensors.size() != 8) {
            MD_LOG_ERROR << "StructureV2Layout should has 8 output tensors,"
                "but got " << static_cast<int>(single_batch_tensors.size()) << " now!" << std::endl;
        }
        // layout_img_info: {image width, image height, resize width, resize height}
        int img_w = layout_img_info[0];
        int img_h = layout_img_info[1];
        int in_w = layout_img_info[2];
        int in_h = layout_img_info[3];
        float scale_factor_w = static_cast<float>(in_w) / static_cast<float>(img_w);
        float scale_factor_h = static_cast<float>(in_h) / static_cast<float>(img_h);

        std::vector<DetectionResult> bbox_results;
        bbox_results.resize(num_class_); // tmp result for each class

        // decode score, label, box
        for (int i = 0; i < fpn_stride_.size(); ++i) {
            int feature_h = std::ceil(static_cast<float>(in_h) / fpn_stride_[i]);
            int feature_w = std::ceil(static_cast<float>(in_w) / fpn_stride_[i]);
            const Tensor& prob_tensor = single_batch_tensors[i];
            const Tensor& bbox_tensor = single_batch_tensors[i + fpn_stride_.size()];
            const float* prob_data = reinterpret_cast<const float*>(prob_tensor.data());
            const float* bbox_data = reinterpret_cast<const float*>(bbox_tensor.data());
            for (int idx = 0; idx < feature_h * feature_w; ++idx) {
                // score and label
                float score = 0.f;
                int label = 0;
                for (int j = 0; j < num_class_; ++j) {
                    if (prob_data[idx * num_class_ + j] > score) {
                        score = prob_data[idx * num_class_ + j];
                        label = j;
                    }
                }
                // bbox
                if (score > score_threshold_) {
                    int row = idx / feature_w;
                    int col = idx % feature_w;
                    std::vector<float> bbox_pred(bbox_data + idx * 4 * reg_max_,
                                                 bbox_data + (idx + 1) * 4 * reg_max_);
                    bbox_results[label].boxes.push_back(dis_pred_to_bbox(
                        bbox_pred, col, row, fpn_stride_[i], in_w, in_h, reg_max_));
                    bbox_results[label].scores.push_back(score);
                    bbox_results[label].label_ids.push_back(label);
                }
            }
        }

        result->clear();
        // nms for per class, i in [0~num_class-1]
        for (int i = 0; i < bbox_results.size(); ++i) {
            if (bbox_results[i].boxes.empty()) {
                continue;
            }
            vision::utils::nms(&bbox_results[i], nms_threshold_);
            // fill output results
            for (int j = 0; j < bbox_results[i].boxes.size(); ++j) {
                result->scores.push_back(bbox_results[i].scores[j]);
                result->label_ids.push_back(bbox_results[i].label_ids[j]);
                result->boxes.push_back(
                    {
                        bbox_results[i].boxes[j][0] / scale_factor_w,
                        bbox_results[i].boxes[j][1] / scale_factor_h,
                        bbox_results[i].boxes[j][2] / scale_factor_w,
                        bbox_results[i].boxes[j][3] / scale_factor_h,
                    });
            }
        }
        return true;
    }

    std::array<float, 4> StructureV2LayoutPostprocessor::dis_pred_to_bbox(
        const std::vector<float>& bbox_pred, int x, int y, int stride, int resize_w,
        int resize_h, int reg_max) {
        float ct_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(stride);
        float ct_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(stride);
        std::vector<float> dis_pred;
        dis_pred.resize(4);
        for (int i = 0; i < 4; i++) {
            std::vector<float> bbox_pred_i(bbox_pred.begin() + i * reg_max,
                                           bbox_pred.begin() + (i + 1) * reg_max);
            std::vector<float> dis_after_sm = ocr::softmax(bbox_pred_i);
            float dis = 0.0f;
            for (int j = 0; j < reg_max; j++) {
                dis += static_cast<float>(j) * dis_after_sm[j];
            }
            dis *= static_cast<float>(stride);
            dis_pred[i] = dis;
        }

        float xmin = std::max(ct_x - dis_pred[0], 0.0f);
        float ymin = std::max(ct_y - dis_pred[1], 0.0f);
        float xmax = std::min(ct_x + dis_pred[2], static_cast<float>(resize_w));
        float ymax = std::min(ct_y + dis_pred[3], static_cast<float>(resize_h));

        return {xmin, ymin, xmax, ymax};
    }
} // namespace modeldeploy::vision::ocr

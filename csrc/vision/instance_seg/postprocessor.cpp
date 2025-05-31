//
// Created by aichao on 2025/4/14.
//

#include <numeric> // 需要包含此头文件
#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/instance_seg/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsSegPostprocessor::UltralyticsSegPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
        mask_threshold_ = 0.5;
    }

    bool UltralyticsSegPostprocessor::run(
        std::vector<Tensor>& tensors, std::vector<DetectionResult>* results,
        const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) const {
        //(1,116,8400)->(1,8400,116)  116=4(xc,yc,w,h)+80(coco 80 classes)+32(mask coefficient)
        tensors[0] = tensors[0].transpose({0, 2, 1}).to_tensor();
        auto mask_nums = tensors[1].shape()[1];
        size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            // store mask information
            std::vector<std::vector<float>> mask_embeddings;
            (*results)[bs].clear();
            (*results)[bs].reserve(static_cast<int>(tensors[0].shape()[1]));
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const float* data = static_cast<const float*>(tensors[0].data()) +
                bs * tensors[0].shape()[1] * tensors[0].shape()[2];
            for (size_t i = 0; i < tensors[0].shape()[1]; ++i) {
                size_t s = i * tensors[0].shape()[2];
                float cls_conf = data[s + 4];
                std::vector mask_embedding(
                    data + s + tensors[0].shape()[2] - mask_nums,
                    data + s + tensors[0].shape()[2]);
                for (float& mask_embedding_el : mask_embedding) {
                    mask_embedding_el *= cls_conf;
                }
                const float* max_class_score = std::max_element(
                    data + s + 4, data + s + tensors[0].shape()[2] - mask_nums);
                float confidence = *max_class_score;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(std::distance(data + s + 4, max_class_score));
                // convert from [x, y, w, h] to [x1, y1, x2, y2]
                (*results)[bs].boxes.emplace_back(
                    std::array{
                        data[s + 0] - data[s + 2] / 2.0f,
                        data[s + 1] - data[s + 3] / 2.0f,
                        data[s + 0] + data[s + 2] / 2.0f,
                        data[s + 1] + data[s + 3] / 2.0f
                    });
                (*results)[bs].label_ids.push_back(label_id);
                (*results)[bs].scores.push_back(confidence);
                mask_embeddings.emplace_back(std::move(mask_embedding));
            }

            if ((*results)[bs].boxes.empty()) {
                continue;
            }
            // get box index after nms
            std::vector<int> indexs;
            utils::nms(&(*results)[bs], nms_threshold_, &indexs);
            // deal with mask
            // step1: MatMul, (box_nums * 32) x (32 * 160 * 160) = box_nums * 160 * 160
            // step2: Sigmoid
            // step3: Resize to original image size
            // step4: Select pixels greater than threshold and crop
            (*results)[bs].contain_masks = true;
            (*results)[bs].masks.resize((*results)[bs].boxes.size());
            const float* data_mask =
                static_cast<const float*>(tensors[1].data()) +
                bs * tensors[1].shape()[1] * tensors[1].shape()[2] * tensors[1].shape()[3];
            auto mask_proto = cv::Mat(static_cast<int>(tensors[1].shape()[1]),
                                      static_cast<int>(tensors[1].shape()[2] * tensors[1].shape()[3]),
                                      CV_32FC(1), const_cast<float*>(data_mask));
            // vector to cv::Mat for MatMul
            // after push_back, Mat of m*n becomes (m + 1) * n
            cv::Mat mask_proposals;
            for (auto index : indexs) {
                auto tmp = cv::Mat(1, mask_nums, CV_32FC(1), mask_embeddings[index].data());
                mask_proposals.push_back(tmp);
            }
            cv::Mat matmul_result = (mask_proposals * mask_proto).t();
            cv::Mat masks = matmul_result.reshape(
                static_cast<int>((*results)[bs].boxes.size()), {
                    static_cast<int>(tensors[1].shape()[2]),
                    static_cast<int>(tensors[1].shape()[3])
                });
            // split for boxes nums
            std::vector<cv::Mat> mask_channels;
            cv::split(masks, mask_channels);
            // scale the boxes to the origin image shape
            auto iter_out = ims_info[bs].find("output_shape");
            auto iter_ipt = ims_info[bs].find("input_shape");
            if (!(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end())) {
                MD_LOG_ERROR << "Cannot find input_shape or output_shape from im_info.";
            }
            float out_h = iter_out->second[0];
            float out_w = iter_out->second[1];
            float ipt_h = iter_ipt->second[0];
            float ipt_w = iter_ipt->second[1];
            float scale = std::min(out_h / ipt_h, out_w / ipt_w);
            float pad_h = (out_h - ipt_h * scale) / 2;
            float pad_w = (out_w - ipt_w * scale) / 2;
            // for mask
            float pad_h_mask = pad_h / out_h * static_cast<float>(tensors[1].shape()[2]);
            float pad_w_mask = pad_w / out_w * static_cast<float>(tensors[1].shape()[3]);
            for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
                auto& box = (*results)[bs].boxes[i];

                // Remove padding and apply scale
                box[0] = (box[0] - pad_w) / scale;
                box[1] = (box[1] - pad_h) / scale;
                box[2] = (box[2] - pad_w) / scale;
                box[3] = (box[3] - pad_h) / scale;

                // Clip to image boundaries
                box[0] = utils::clamp(box[0], 0.0f, ipt_w);
                box[1] = utils::clamp(box[1], 0.0f, ipt_h);
                box[2] = utils::clamp(box[2], 0.0f, ipt_w);
                box[3] = utils::clamp(box[3], 0.0f, ipt_h);

                // deal with mask
                cv::Mat dest, mask;
                // sigmoid
                cv::exp(-mask_channels[i], dest);
                dest = 1.0 / (1.0 + dest);
                // crop mask for feature map
                int x1 = static_cast<int>(pad_w_mask);
                int y1 = static_cast<int>(pad_h_mask);
                int x2 = static_cast<int>(static_cast<float>(tensors[1].shape()[3]) - pad_w_mask);
                int y2 = static_cast<int>(static_cast<float>(tensors[1].shape()[2]) - pad_h_mask);
                cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                dest = dest(roi);
                cv::resize(dest, mask,
                           {static_cast<int>(ipt_w), static_cast<int>(ipt_h)}, 0, 0,
                           cv::INTER_LINEAR);
                // crop mask for source img
                int x1_src = static_cast<int>(round((*results)[bs].boxes[i][0]));
                int y1_src = static_cast<int>(round((*results)[bs].boxes[i][1]));
                int x2_src = static_cast<int>(round((*results)[bs].boxes[i][2]));
                int y2_src = static_cast<int>(round((*results)[bs].boxes[i][3]));
                cv::Rect roi_src(x1_src, y1_src, x2_src - x1_src, y2_src - y1_src);
                mask = mask(roi_src);
                mask = mask > mask_threshold_;
                // save mask in DetectionResult
                int keep_mask_h = y2_src - y1_src;
                int keep_mask_w = x2_src - x1_src;
                int keep_mask_num_el = keep_mask_h * keep_mask_w;
                (*results)[bs].masks[i].resize(keep_mask_num_el);
                (*results)[bs].masks[i].shape = {keep_mask_h, keep_mask_w};
                auto* keep_mask_ptr = static_cast<uint8_t*>((*results)[bs].masks[i].data());
                std::copy_n(mask.ptr(), keep_mask_num_el, keep_mask_ptr);
            }
        }
        return true;
    }
}

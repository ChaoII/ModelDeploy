//
// Created by aichao on 2025/4/14.
//

#include <numeric> // 需要包含此头文件
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/iseg/postprocessor.h"

namespace modeldeploy::vision::detection {
    UltralyticsSegPostprocessor::UltralyticsSegPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
        mask_threshold_ = 0.5;
    }

    bool UltralyticsSegPostprocessor::run_without_nms(
        std::vector<Tensor>& tensors, std::vector<std::vector<InstanceSegResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        //(1,116,8400)->(1,8400,116)  116=4(xc,yc,w,h)+80(coco 80 classes)+32(mask coefficient)
        tensors[0] = tensors[0].transpose({0, 2, 1}).to_tensor();
        auto mask_nums = tensors[1].shape()[1];
        size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            // store mask information
            std::vector<std::vector<float>> mask_embeddings;
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //8400
            const size_t dim2 = tensors[0].shape()[2]; //84
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<InstanceSegResult> _results;
            _results.reserve(dim1); // 预分配空间
            mask_embeddings.reserve(dim1); // 预分配空间
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                float cls_conf = attr_ptr[4];
                // 使用引用避免拷贝
                const std::vector<float> mask_embedding(attr_ptr + dim2 - mask_nums, attr_ptr + dim2);
                // 直接在原vector上操作，避免创建临时变量
                for (size_t j = 0; j < mask_embedding.size(); ++j) {
                    const_cast<std::vector<float>&>(mask_embedding)[j] *= cls_conf;
                }
                const float* max_class_score = std::max_element(attr_ptr + 4, attr_ptr + dim2 - mask_nums);
                float confidence = *max_class_score;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(std::distance(attr_ptr + 4, max_class_score));
                // convert from [xc, yc, w, h] to [w, y, w, h]
                Rect2f box{
                    attr_ptr[0] - attr_ptr[2] / 2.0f,
                    attr_ptr[1] - attr_ptr[3] / 2.0f,
                    attr_ptr[2],
                    attr_ptr[3]
                };
                mask_embeddings.push_back(mask_embedding);
                _results.push_back({box, Mask(), label_id, confidence}); // 使用emplace_back避免拷贝
            }
            if (_results.empty()) {
                continue;
            }
            // get box index after nms
            std::vector<int> indexs;
            utils::nms(&_results, nms_threshold_, &indexs);
            // deal with mask
            // step1: MatMul, (box_nums * 32) x (32 * 160 * 160) = box_nums * 160 * 160
            // step2: Sigmoid
            // step3: Resize to original image size
            // step4: Select pixels greater than threshold and crop

            // (32,160,160)
            int mask_c = tensors[1].shape()[1];
            int mask_h = tensors[1].shape()[2];
            int mask_w = tensors[1].shape()[3];

            float* data_mask = static_cast<float*>(tensors[1].data()) + bs * mask_c * mask_h * mask_w;
            // 直接使用数据指针创建Mat，避免拷贝
            cv::Mat mask_proto(mask_c, mask_h * mask_w, CV_32FC1, data_mask);
            // vector to cv::Mat for MatMul
            // n * 32
            const int num_instances = static_cast<int>(indexs.size()); //n
            // 预分配Mat内存
            cv::Mat mask_proposals(num_instances, mask_nums, CV_32FC1);
            for (int i = 0; i < num_instances; ++i) {
                auto* dst_ptr = mask_proposals.ptr<float>(i);
                const auto& embedding = mask_embeddings[indexs[i]];
                std::memcpy(dst_ptr, embedding.data(), mask_nums * sizeof(float));
            }
            // (n,32) @ (32,160*160) = (n,160*160)
            // 使用更高效的矩阵乘法
            cv::Mat matmul_result;
            cv::gemm(mask_proposals, mask_proto, 1.0, cv::Mat(), 0.0, matmul_result);
            // (n,160,160)
            cv::Mat masks = matmul_result.t();
            masks = masks.reshape(static_cast<int>(_results.size()), {mask_w, mask_h});
            // split for boxes nums
            std::vector<cv::Mat> mask_channels;
            mask_channels.reserve(_results.size());
            cv::split(masks, mask_channels);
            // scale the boxes to the origin image shape
            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float out_h = letter_box_records[bs].out_h;
            const float out_w = letter_box_records[bs].out_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            // for mask
            float pad_h_mask = pad_h / out_h * static_cast<float>(tensors[1].shape()[2]);
            float pad_w_mask = pad_w / out_w * static_cast<float>(tensors[1].shape()[3]);
            for (size_t i = 0; i < _results.size(); ++i) {
                auto& box = _results[i].box;

                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;

                // 限制在图像边界内
                x1 = utils::clamp(x1, 0.0f, ipt_w);
                y1 = utils::clamp(y1, 0.0f, ipt_h);
                x2 = utils::clamp(x2, 0.0f, ipt_w);
                y2 = utils::clamp(y2, 0.0f, ipt_h);

                // 重新赋值到 box
                box.x = std::round(x1);
                box.y = std::round(y1);
                box.width = std::round(x2 - x1);
                box.height = std::round(y2 - y1);
                // deal with mask
                cv::Mat dest, mask;
                // 使用更高效的sigmoid实现
                cv::exp(-mask_channels[i], dest);
                dest = 1.0 / (1.0 + dest);
                // crop mask for feature map
                int _x1 = static_cast<int>(pad_w_mask);
                int _y1 = static_cast<int>(pad_h_mask);
                int _x2 = static_cast<int>(tensors[1].shape()[3] - pad_w_mask);
                int _y2 = static_cast<int>(tensors[1].shape()[2] - pad_h_mask);
                cv::Rect roi(_x1, _y1, _x2 - _x1, _y2 - _y1);
                dest = dest(roi);
                cv::resize(dest, mask, cv::Size2f{ipt_w, ipt_h}, 0, 0, cv::INTER_LINEAR);
                mask = mask(utils::rect2f_to_cv_type(box));
                mask = mask > mask_threshold_;
                // save mask in DetectionResult
                int keep_mask_h = static_cast<int>(box.height);
                int keep_mask_w = static_cast<int>(box.width);
                int keep_mask_num_el = keep_mask_h * keep_mask_w;
                _results[i].mask.resize(keep_mask_num_el);
                _results[i].mask.shape = {keep_mask_h, keep_mask_w};
                auto* keep_mask_ptr = static_cast<uint8_t*>(_results[i].mask.data());
                // 使用memcpy提高拷贝效率
                std::memcpy(keep_mask_ptr, mask.ptr(), keep_mask_num_el);
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsSegPostprocessor::run_with_nms(
        std::vector<Tensor>& tensors, std::vector<std::vector<InstanceSegResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        //(1,300,38)38=4(xc,yc,w,h)+1(score)+1(label_id)+32(mask coefficient)
        auto mask_nums = tensors[1].shape()[1];
        size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            // store mask information
            std::vector<std::vector<float>> mask_embeddings;
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //300
            const size_t dim2 = tensors[0].shape()[2]; //38
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<InstanceSegResult> _results;
            _results.reserve(dim1); // 预分配空间
            mask_embeddings.reserve(dim1); // 预分配空间
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                float cls_conf = attr_ptr[4];
                // 使用引用避免拷贝
                const std::vector<float> mask_embedding(attr_ptr + dim2 - mask_nums, attr_ptr + dim2);
                // 直接在原vector上操作，避免创建临时变量
                for (size_t j = 0; j < mask_embedding.size(); ++j) {
                    const_cast<std::vector<float>&>(mask_embedding)[j] *= cls_conf;
                }
                float score = attr_ptr[4];
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) {
                    continue;
                }
                int32_t label_id = attr_ptr[5];
                // convert from [x1, y1, x2, y2] to [w, y, w, h]
                Rect2f box{
                    attr_ptr[0],
                    attr_ptr[1],
                    attr_ptr[2] - attr_ptr[0],
                    attr_ptr[3] - attr_ptr[1]
                };
                mask_embeddings.push_back(mask_embedding);
                _results.push_back({box, Mask(), label_id, score}); // 使用emplace_back避免拷贝
            }
            if (_results.empty()) {
                continue;
            }

            // deal with mask
            // step1: MatMul, (box_nums * 32) x (32 * 160 * 160) = box_nums * 160 * 160
            // step2: Sigmoid
            // step3: Resize to original image size
            // step4: Select pixels greater than threshold and crop

            // (32,160,160)
            int mask_c = tensors[1].shape()[1];
            int mask_h = tensors[1].shape()[2];
            int mask_w = tensors[1].shape()[3];

            float* data_mask = static_cast<float*>(tensors[1].data()) + bs * mask_c * mask_h * mask_w;
            // 直接使用数据指针创建Mat，避免拷贝
            cv::Mat mask_proto(mask_c, mask_h * mask_w, CV_32FC1, data_mask);
            // vector to cv::Mat for MatMul
            // n * 32
            const int num_instances = static_cast<int>(_results.size()); //n
            // 预分配Mat内存
            cv::Mat mask_proposals(num_instances, mask_nums, CV_32FC1);
            for (int i = 0; i < num_instances; ++i) {
                auto* dst_ptr = mask_proposals.ptr<float>(i);
                const auto& embedding = mask_embeddings[i];
                std::memcpy(dst_ptr, embedding.data(), mask_nums * sizeof(float));
            }
            // (n,32) @ (32,160*160) = (n,160*160)
            // 使用更高效的矩阵乘法
            cv::Mat matmul_result;
            cv::gemm(mask_proposals, mask_proto, 1.0, cv::Mat(), 0.0, matmul_result);
            // (n,160,160)
            cv::Mat masks = matmul_result.t();
            masks = masks.reshape(static_cast<int>(_results.size()), {mask_w, mask_h});

            //split for boxes nums
            std::vector<cv::Mat> mask_channels;
            mask_channels.reserve(_results.size());
            cv::split(masks, mask_channels);

            // scale the boxes to the origin image shape
            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float out_h = letter_box_records[bs].out_h;
            const float out_w = letter_box_records[bs].out_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            // for mask
            float pad_h_mask = pad_h / out_h * static_cast<float>(tensors[1].shape()[2]);
            float pad_w_mask = pad_w / out_w * static_cast<float>(tensors[1].shape()[3]);
            for (size_t i = 0; i < _results.size(); ++i) {
                auto& box = _results[i].box;

                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;

                // 限制在图像边界内
                x1 = utils::clamp(x1, 0.0f, ipt_w);
                y1 = utils::clamp(y1, 0.0f, ipt_h);
                x2 = utils::clamp(x2, 0.0f, ipt_w);
                y2 = utils::clamp(y2, 0.0f, ipt_h);

                // 重新赋值到 box
                box.x = std::round(x1);
                box.y = std::round(y1);
                box.width = std::round(x2 - x1);
                box.height = std::round(y2 - y1);
                // deal with mask
                cv::Mat dest, mask;
                // 使用更高效的sigmoid实现
                cv::exp(-mask_channels[i], dest);
                dest = 1.0 / (1.0 + dest);
                // crop mask for feature map
                int _x1 = static_cast<int>(pad_w_mask);
                int _y1 = static_cast<int>(pad_h_mask);
                int _x2 = static_cast<int>(tensors[1].shape()[3] - pad_w_mask);
                int _y2 = static_cast<int>(tensors[1].shape()[2] - pad_h_mask);
                cv::Rect roi(_x1, _y1, _x2 - _x1, _y2 - _y1);
                dest = dest(roi);
                cv::resize(dest, mask, cv::Size2f{ipt_w, ipt_h}, 0, 0, cv::INTER_LINEAR);
                mask = mask(utils::rect2f_to_cv_type(box));
                mask = mask > mask_threshold_;
                // save mask in DetectionResult
                int keep_mask_h = static_cast<int>(box.height);
                int keep_mask_w = static_cast<int>(box.width);
                int keep_mask_num_el = keep_mask_h * keep_mask_w;
                _results[i].mask.resize(keep_mask_num_el);
                _results[i].mask.shape = {keep_mask_h, keep_mask_w};
                auto* keep_mask_ptr = static_cast<uint8_t*>(_results[i].mask.data());
                // 使用memcpy提高拷贝效率
                std::memcpy(keep_mask_ptr, mask.ptr(), keep_mask_num_el);
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsSegPostprocessor::run(std::vector<Tensor>& tensors,
                                          std::vector<std::vector<InstanceSegResult>>* results,
                                          const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors[0].shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor." << std::endl;
            return false;
        }
        if (tensors[0].shape()[2] == 38) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}

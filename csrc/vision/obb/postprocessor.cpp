//
// Created by aichao on 2025/5/30.
//

#include <numeric>
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/obb/postprocessor.h"

#include <utils/utils.h>

namespace modeldeploy::vision::detection {
    UltralyticsObbPostprocessor::UltralyticsObbPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
    }

    bool UltralyticsObbPostprocessor::run_without_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<ObbResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        //  [B, C, N] 布局：C=20 = 4(xc,yc,w,h)+classes_num(15)+1(angle)，N=21504。
        //  无需转置物化：按 anchor 分块顺序扫各 class 行求 max，只对过阈候选解码。
        const size_t channels = tensors[0].shape()[1];  // 20
        const size_t anchors = tensors[0].shape()[2];   // 21504
        const size_t num_classes = channels - 5;        // 15（通道 4..18），angle 在最后通道 dim-1
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * channels * anchors;
            const size_t angle_channel = channels - 1;  // 19
            std::vector<ObbResult> _results;
            // 分块：块内锚点顺序可达，class 行顺序读（cache 友好），无全量转置写
            const size_t kBlock = 256;
            for (size_t blk = 0; blk < anchors; blk += kBlock) {
                const size_t cnt = std::min(kBlock, anchors - blk);
                float maxs[256];
                int argmax[256];
                // 初始化：首个 class 通道（通道4）作为初值
                const float* c0 = data + 4 * anchors + blk;
                for (size_t i = 0; i < cnt; ++i) { maxs[i] = c0[i]; argmax[i] = 0; }
                for (size_t c = 1; c < num_classes; ++c) {
                    const float* row = data + (4 + c) * anchors + blk;
                    for (size_t i = 0; i < cnt; ++i) {
                        if (row[i] > maxs[i]) { maxs[i] = row[i]; argmax[i] = static_cast<int>(c); }
                    }
                }
                for (size_t i = 0; i < cnt; ++i) {
                    const float confidence = maxs[i];
                    if (confidence <= conf_threshold_) continue;
                    const size_t a = blk + i;
                    // convert from [xc, yc, w, h, a]
                    // 其中a为angle矩形框的旋转角度, 默认为弧度制(但是OpenCV的RotatedRect的旋转角度，默认为角度制)
                    RotatedRect rotated_boxes = {
                        data[0 * anchors + a], data[1 * anchors + a],
                        data[2 * anchors + a], data[3 * anchors + a],
                        data[angle_channel * anchors + a] * 180 / 3.141592653f
                    };
                    _results.push_back({rotated_boxes, argmax[i], confidence});
                }
            }
            if (_results.empty()) {
                continue;
            }
            utils::obb_nms(&_results, nms_threshold_);
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            for (auto& result : _results) {
                auto& box = result.rotated_box;
                // clip box()
                //先减去 padding,再除以缩放因子scale;
                box.xc = (box.xc - pad_w) / scale;
                box.yc = (box.yc - pad_h) / scale;
                box.width = box.width / scale;
                box.height = box.height / scale;
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsObbPostprocessor::run_with_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<ObbResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        // transpose(1,300,7)(xc, yc, w, h, score, label_id, angle)
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            // 官方模型为300
            const auto dim1 = tensors[0].shape()[1];
            // 官方模型为7 (xc, yc, w, h, score, label_id, angle)
            const auto dim2 = tensors[0].shape()[2];
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<ObbResult> _results;
            for (size_t i = 0; i < dim1; ++i) {
                const auto attr_ptr = data + i * dim2;
                float score = attr_ptr[4];
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) {
                    continue;
                }
                auto label_id = static_cast<int32_t>(attr_ptr[5]);
                // convert from [xc, yc, w, h, score, label_id, angle]
                // 带 NMS 输出布局 (dim2=7)：angle 在索引 6（弧度制）
                // OpenCV 的 RotatedRect 旋转角度默认角度制
                RotatedRect rotated_boxes = {
                    attr_ptr[0], attr_ptr[1],
                    attr_ptr[2], attr_ptr[3],
                    attr_ptr[6] * 180 / 3.141592653f
                };
                _results.push_back({rotated_boxes, label_id, score});
            }
            if (_results.empty()) {
                continue;
            }
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            for (auto& result : _results) {
                auto& box = result.rotated_box;
                // clip box()
                //先减去 padding,再除以缩放因子scale;
                box.xc = (box.xc - pad_w) / scale;
                box.yc = (box.yc - pad_h) / scale;
                box.width = box.width / scale;
                box.height = box.height / scale;
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }

    bool UltralyticsObbPostprocessor::run(std::vector<Tensor>& tensors,
                                          std::vector<std::vector<ObbResult>>* results,
                                          const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors[0].shape().size() == 2) {
            // ncnn batch==1 压掉首维；用通用 Tensor::expand_dim(0) 补回 batch 维后落体。
            tensors[0].expand_dim(0);
        }
        if (tensors[0].shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor." << std::endl;
            return false;
        }
        if (tensors[0].shape()[2] == 7) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}

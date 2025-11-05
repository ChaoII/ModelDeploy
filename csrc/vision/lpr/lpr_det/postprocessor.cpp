//
// Created by aichao on 2025/6/10.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "utils/utils.h"
#include "vision/lpr/lpr_det/postprocessor.h"


namespace modeldeploy::vision::lpr {
    LprDetPostprocessor::LprDetPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
        landmarks_per_card_ = 4;
    }

    bool LprDetPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<std::vector<KeyPointsResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //25200
            const size_t dim2 = tensors[0].shape()[2]; //15
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<KeyPointsResult> _results;
            _results.reserve(dim1);
            // x,y,w,h,obj_conf,x1,y1,x2,y2,x3,y3,x4,y4,cls_conf0(单层车牌),cls_conf1(双层车牌)
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                const float obj_conf = attr_ptr[4];
                // const float cls_conf = reg_cls_ptr[13];
                // 0: 单层车牌
                // 1: 双层车牌
                std::vector<float> cls_confs = {attr_ptr + 13, attr_ptr + dim2};
                const int class_id = argmax(cls_confs);
                const auto cls_conf = *std::max_element(cls_confs.begin(), cls_confs.end());
                float confidence = obj_conf * cls_conf;
                // filter boxes by conf_threshold
                if (confidence <= conf_threshold_) {
                    continue;
                }

                const float x = attr_ptr[0];
                const float y = attr_ptr[1];
                const float w = attr_ptr[2];
                const float h = attr_ptr[3];

                std::vector<Point3f> landmarks;
                landmarks.reserve(landmarks_per_card_);
                // decode landmarks (default 5 landmarks)
                if (landmarks_per_card_ > 0) {
                    const float* landmarks_ptr = attr_ptr + 5;
                    for (size_t j = 0; j < landmarks_per_card_ * 2; j += 2) {
                        landmarks.emplace_back(
                            landmarks_ptr[j], landmarks_ptr[j + 1], 0.0f);
                    }
                }
                _results.push_back({Rect2f{x - w / 2.f, y - h / 2.f, w, h}, landmarks, class_id, confidence});
            }
            if (_results.empty()) {
                continue;
            }
            utils::nms(&_results, nms_threshold_);
            // scale the boxes to the origin image shape

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                auto& box = result.box;
                auto& landmarks = result.keypoints;
                // clip box()
                //1 先减去 padding;2除以缩放因子scale 3最后限制在原始图像范围内 [0, width], [0, height]。
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
                box.x = std::roundf(x1);
                box.y = std::roundf(y1);
                box.width = std::roundf(x2 - x1);
                box.height = std::roundf(y2 - y1);


                // scale and clip landmarks
                for (auto& landmark : landmarks) {
                    landmark.x = std::max((landmark.x - pad_w) / scale, 0.0f);
                    landmark.y = std::max((landmark.y - pad_h) / scale, 0.0f);

                    landmark.x = std::min(landmark.x, ipt_w - 1.0f);
                    landmark.y = std::min(landmark.y, ipt_h - 1.0f);
                }
            }
            results->at(bs) = std::move(_results);
        }
        return true;
    }
}

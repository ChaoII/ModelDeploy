#include <algorithm>
#include <cstring>
#include <cmath>
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/sam/postprocessor.h"

namespace modeldeploy::vision::seg {
    constexpr size_t kFastSamMaskNums = 32;  // mask 系数维度

    FastSamPostprocessor::FastSamPostprocessor() {
        conf_threshold_ = 0.30f;
        nms_threshold_ = 0.40f;
        mask_threshold_ = 0.5f;
    }

    bool FastSamPostprocessor::run(
        std::vector<Tensor>& tensors, std::vector<std::vector<InstanceSegResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors.size() < 2) {
            MD_LOG_ERROR << "FastSAM expects 2 outputs (box head + mask proto)." << std::endl;
            return false;
        }
        if (tensors[0].dtype() != DataType::FP32 || tensors[1].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        // tensors[0]: [B, 37, N]  YOLO 式布局:37 = 4(xc,yc,w,h) + 1(score) + 32(mask 系数)，
        //             N=8400 候选框放在最后一维
        // tensors[1]: [B, 32, H, W]  mask prototype
        const auto& s0 = tensors[0].shape();
        const auto& s1 = tensors[1].shape();
        if (s0.size() != 3 || s1.size() != 4) {
            MD_LOG_ERROR << "Unexpected FastSAM output ranks (expect [B,37,N] + [B,32,H,W])."
                         << std::endl;
            return false;
        }
        const size_t batch = s0[0];
        const size_t channels = s0[1];  // 37
        const size_t anchors = s0[2];   // 8400
        const size_t num_box_channels = 4;
        if (channels < num_box_channels + 1 + kFastSamMaskNums) {
            MD_LOG_ERROR << "Unexpected box-head channels: " << channels << std::endl;
            return false;
        }
        const int mask_c = static_cast<int>(s1[1]);
        const int mask_h = static_cast<int>(s1[2]);
        const int mask_w = static_cast<int>(s1[3]);
        if (static_cast<size_t>(mask_c) != kFastSamMaskNums) {
            MD_LOG_ERROR << "Unexpected mask proto channels: " << mask_c << std::endl;
            return false;
        }

        auto& values = *results;
        values.resize(batch);
        const float* data0 = static_cast<const float*>(tensors[0].data());
        for (size_t bs = 0; bs < batch; ++bs) {
            std::vector<std::vector<float>> mask_embeddings;
            std::vector<InstanceSegResult> _results;
            const float* data = data0 + bs * channels * anchors;
            for (size_t a = 0; a < anchors; ++a) {
                // xc, yc, w, h（相对 640 输入，需按 letterbox 回退到原图）
                const float xc = data[0 * anchors + a];
                const float yc = data[1 * anchors + a];
                const float w  = data[2 * anchors + a];
                const float h  = data[3 * anchors + a];
                const float score = data[4 * anchors + a];
                if (score <= conf_threshold_) continue;
                Rect2f box{xc - w / 2.0f, yc - h / 2.0f, w, h};
                std::vector<float> embed(kFastSamMaskNums);
                // 与原版 ultralytics 语义一致:mask 系数取原始值(pred[:, 6:])，不乘 score
                for (size_t j = 0; j < kFastSamMaskNums; ++j) {
                    embed[j] = data[(5 + j) * anchors + a];
                }
                mask_embeddings.push_back(std::move(embed));
                _results.push_back({box, Mask(), 0, score});
            }
            if (_results.empty()) continue;

            std::vector<int> indexs;
            utils::nms(&_results, nms_threshold_, &indexs);
            const int num_instances = static_cast<int>(indexs.size());

            // mask = 系数( n×32 ) @ proto( 32×H*W )
            cv::Mat mask_proto(mask_c, mask_h * mask_w, CV_32FC1,
                               static_cast<float*>(tensors[1].data()) + bs * mask_c * mask_h * mask_w);
            cv::Mat mask_proposals(num_instances, static_cast<int>(kFastSamMaskNums), CV_32FC1);
            for (int i = 0; i < num_instances; ++i) {
                std::memcpy(mask_proposals.ptr<float>(i), mask_embeddings[indexs[i]].data(),
                            kFastSamMaskNums * sizeof(float));
            }
            cv::Mat matmul_result;
            cv::gemm(mask_proposals, mask_proto, 1.0, cv::Mat(), 0.0, matmul_result);

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float out_h = letter_box_records[bs].out_h;
            const float out_w = letter_box_records[bs].out_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;
            const float pad_h_mask = pad_h / out_h * static_cast<float>(mask_h);
            const float pad_w_mask = pad_w / out_w * static_cast<float>(mask_w);

            // 注意:utils::nms 原地重建 _results(按分数降序、仅保留),故此处直接用 _results[i]，
            //     mask 系数仍按原始索引 indexs[i] 取(mask_embeddings 未被 nms 重排)。
            for (int i = 0; i < num_instances; ++i) {
                auto& box = _results[i].box;
                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;
                x1 = std::clamp(x1, 0.0f, ipt_w);
                y1 = std::clamp(y1, 0.0f, ipt_h);
                x2 = std::clamp(x2, 0.0f, ipt_w);
                y2 = std::clamp(y2, 0.0f, ipt_h);
                box.x = std::round(x1);
                box.y = std::round(y1);
                box.width = std::round(x2 - x1);
                box.height = std::round(y2 - y1);

                const cv::Mat mask_channel = matmul_result.row(i).reshape(1, mask_h);
                const int _x1 = static_cast<int>(pad_w_mask);
                const int _y1 = static_cast<int>(pad_h_mask);
                const int _x2 = static_cast<int>(mask_w - pad_w_mask);
                const int _y2 = static_cast<int>(mask_h - pad_h_mask);
                const float fw = static_cast<float>(_x2 - _x1);
                const float fh = static_cast<float>(_y2 - _y1);
                const float bx1 = std::clamp(box.x, 0.0f, ipt_w);
                const float by1 = std::clamp(box.y, 0.0f, ipt_h);
                const float bx2 = std::clamp(box.x + box.width, 0.0f, ipt_w);
                const float by2 = std::clamp(box.y + box.height, 0.0f, ipt_h);
                const int mx1 = static_cast<int>(_x1 + bx1 / ipt_w * fw);
                const int my1 = static_cast<int>(_y1 + by1 / ipt_h * fh);
                const int mx2 = static_cast<int>(_x1 + bx2 / ipt_w * fw);
                const int my2 = static_cast<int>(_y1 + by2 / ipt_h * fh);
                cv::Mat dest, mask;
                if (mx2 > mx1 && my2 > my1 && box.width > 0 && box.height > 0) {
                    cv::Rect box_roi(mx1, my1, mx2 - mx1, my2 - my1);
                    dest = mask_channel(box_roi);
                    cv::exp(-dest, dest);
                    dest = 1.0 / (1.0 + dest);
                    cv::resize(dest, mask,
                               cv::Size(static_cast<int>(box.width), static_cast<int>(box.height)),
                               0, 0, cv::INTER_LINEAR);
                    mask = mask > mask_threshold_;
                } else {
                    mask = cv::Mat::zeros(static_cast<int>(box.height),
                                          static_cast<int>(box.width), CV_8UC1);
                }
                const int kh = static_cast<int>(box.height);
                const int kw = static_cast<int>(box.width);
                _results[i].mask.resize(kh * kw);
                _results[i].mask.shape = {kh, kw};
                if (kh > 0 && kw > 0) {
                    std::memcpy(static_cast<uint8_t*>(_results[i].mask.data()), mask.ptr(),
                                static_cast<size_t>(kh) * kw);
                }
            }
            values[bs] = std::move(_results);
        }
        return true;
    }
} // namespace modeldeploy::vision::seg

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

            // 原版语义(process_mask_native, retina 路径):把 160x160 mask 上采样到原图尺寸,
            // 按原图 box 裁剪,保证 mask 贴合物体轮廓而非 proto 低分辨率放大后的粗块。
            const int fh = static_cast<int>(ipt_h);
            const int fw = static_cast<int>(ipt_w);
            for (int i = 0; i < num_instances; ++i) {
                auto& box = _results[i].box;
                // box(640 坐标) -> 原图坐标
                float ox1 = (box.x - pad_w) / scale;
                float oy1 = (box.y - pad_h) / scale;
                float ox2 = (box.x + box.width - pad_w) / scale;
                float oy2 = (box.y + box.height - pad_h) / scale;
                ox1 = std::clamp(ox1, 0.0f, ipt_w);
                oy1 = std::clamp(oy1, 0.0f, ipt_h);
                ox2 = std::clamp(ox2, 0.0f, ipt_w);
                oy2 = std::clamp(oy2, 0.0f, ipt_h);
                box.x = std::round(ox1);
                box.y = std::round(oy1);
                box.width = std::round(ox2 - ox1);
                box.height = std::round(oy2 - oy1);

                // 160x160 -> 原图 双线性上采样 + sigmoid
                cv::Mat feat = matmul_result.row(i).reshape(1, mask_h);
                cv::Mat feat_up;
                cv::resize(feat, feat_up, cv::Size(fw, fh), 0, 0, cv::INTER_LINEAR);
                cv::exp(-feat_up, feat_up);
                feat_up = 1.0 / (1.0 + feat_up);  // sigmoid

                cv::Mat mask;
                if (box.width > 0 && box.height > 0) {
                    cv::Rect roi(static_cast<int>(box.x), static_cast<int>(box.y),
                                 static_cast<int>(box.width), static_cast<int>(box.height));
                    roi &= cv::Rect(0, 0, fw, fh);
                    cv::Mat cropped = feat_up(roi);
                    mask = cropped > mask_threshold_;
                } else {
                    mask = cv::Mat::zeros(0, 0, CV_8UC1);
                }
                const int kh = mask.rows;
                const int kw = mask.cols;
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

//
// insightface buffalo_l det_10g 后处理实现。
// det_10g 输出为 2D shape（如 [12800,1]），与标准 Scrfd 的 3D shape（[1,12800,1]）不同，
// 故独立实现 SCRFD 解码（stride 8/16/32 双 anchor + distance2bbox/kps + NMS）。
//
#include "core/md_log.h"
#include "vision/face/insightface/scrfd/insightface_scrfd_postprocessor.h"
#include <algorithm>

namespace modeldeploy::vision::face {

    namespace {
        constexpr int kFmc = 3;
        constexpr int kStride[3] = {8, 16, 32};
        constexpr int kNumAnchors = 2;

        // anchor centers：交错布局（python np.stack([centers]*2, axis=1).reshape(-1,2)）
        std::vector<float> gen_anchor_centers(int height, int width, int stride) {
            const int K = height * width;
            std::vector<float> base(static_cast<size_t>(K) * 2);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int idx = y * width + x;
                    base[idx * 2] = static_cast<float>(x * stride);
                    base[idx * 2 + 1] = static_cast<float>(y * stride);
                }
            std::vector<float> dup(static_cast<size_t>(K) * kNumAnchors * 2);
            for (int i = 0; i < K; ++i)
                for (int a = 0; a < kNumAnchors; ++a) {
                    dup[(i * kNumAnchors + a) * 2] = base[i * 2];
                    dup[(i * kNumAnchors + a) * 2 + 1] = base[i * 2 + 1];
                }
            return dup;
        }

        std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                             const std::vector<float>& scores, float thresh) {
            const int n = static_cast<int>(boxes.size());
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            std::vector<int> keep;
            std::vector<bool> removed(n, false);
            for (int oi = 0; oi < n; ++oi) {
                const int i = order[oi];
                if (removed[i]) continue;
                keep.push_back(i);
                for (int oj = oi + 1; oj < n; ++oj) {
                    const int j = order[oj];
                    if (removed[j]) continue;
                    const float x1 = std::max(boxes[i][0], boxes[j][0]);
                    const float y1 = std::max(boxes[i][1], boxes[j][1]);
                    const float x2 = std::min(boxes[i][2], boxes[j][2]);
                    const float y2 = std::min(boxes[i][3], boxes[j][3]);
                    const float w = std::max(0.0f, x2 - x1 + 1);
                    const float h = std::max(0.0f, y2 - y1 + 1);
                    const float inter = w * h;
                    const float area_i = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1);
                    const float area_j = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1);
                    const float ovr = inter / (area_i + area_j - inter);
                    if (ovr > thresh) removed[j] = true;
                }
            }
            return keep;
        }
    } // namespace

    bool InsightFaceDetPostprocessor::run(const std::vector<Tensor>& infer_results,
                                          const std::vector<LetterBoxRecord>& letter_box_records,
                                          std::vector<std::vector<InsightFaceBox>>* results) {
        const int dst_w = 640;
        const int dst_h = 640;
        results->resize(letter_box_records.size());
        for (size_t b = 0; b < letter_box_records.size(); ++b) {
            const float det_scale = letter_box_records[b].scale;
            auto& out = (*results)[b];
            out.clear();

            std::vector<float> scores_list, bboxes_list, kpss_list;
            for (int idx = 0; idx < kFmc; ++idx) {
                const int stride = kStride[idx];
                const float* score_ptr = static_cast<const float*>(infer_results[idx].data());
                const float* bbox_ptr = static_cast<const float*>(infer_results[idx + kFmc].data());
                const float* kps_ptr = static_cast<const float*>(infer_results[idx + kFmc * 2].data());
                // 输出 2D shape：[N, ...]；N = num_points（det_10g 无 batch 维）
                const int total = static_cast<int>(infer_results[idx].shape()[0]);
                const int H = dst_h / stride;
                const int W = dst_w / stride;
                if (total != H * W * kNumAnchors) {
                    MD_LOG_ERROR << "unexpected num_points " << total << " for stride " << stride << std::endl;
                    return false;
                }
                // bbox/kps 乘 stride
                std::vector<float> bbox_scaled(static_cast<size_t>(total) * 4);
                std::vector<float> kps_scaled(static_cast<size_t>(total) * 10);
                for (int i = 0; i < total; ++i) {
                    for (int j = 0; j < 4; ++j) bbox_scaled[i * 4 + j] = bbox_ptr[i * 4 + j] * stride;
                    for (int j = 0; j < 10; ++j) kps_scaled[i * 10 + j] = kps_ptr[i * 10 + j] * stride;
                }
                const auto centers = gen_anchor_centers(H, W, stride);
                // distance2bbox + distance2kps
                std::vector<float> bboxes(static_cast<size_t>(total) * 4);
                std::vector<float> kpss(static_cast<size_t>(total) * 10);
                for (int i = 0; i < total; ++i) {
                    const float cx = centers[i * 2], cy = centers[i * 2 + 1];
                    bboxes[i * 4 + 0] = cx - bbox_scaled[i * 4 + 0];
                    bboxes[i * 4 + 1] = cy - bbox_scaled[i * 4 + 1];
                    bboxes[i * 4 + 2] = cx + bbox_scaled[i * 4 + 2];
                    bboxes[i * 4 + 3] = cy + bbox_scaled[i * 4 + 3];
                    for (int j = 0; j < 10; j += 2) {
                        kpss[i * 10 + j] = cx + kps_scaled[i * 10 + j];
                        kpss[i * 10 + j + 1] = cy + kps_scaled[i * 10 + j + 1];
                    }
                }
                for (int i = 0; i < total; ++i) {
                    if (score_ptr[i] < 0.5f) continue;
                    scores_list.push_back(score_ptr[i]);
                    for (int j = 0; j < 4; ++j) bboxes_list.push_back(bboxes[i * 4 + j]);
                    for (int j = 0; j < 10; ++j) kpss_list.push_back(kpss[i * 10 + j]);
                }
            }

            if (scores_list.empty()) continue;
            const int n_det = static_cast<int>(scores_list.size());
            std::vector<int> order(n_det);
            for (int i = 0; i < n_det; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return scores_list[a] > scores_list[b]; });

            std::vector<std::array<float, 4>> boxes_in;
            std::vector<float> scores_in;
            std::vector<float> kpss_in(static_cast<size_t>(n_det) * 10);
            for (int i = 0; i < n_det; ++i) {
                const int o = order[i];
                boxes_in.push_back({bboxes_list[o * 4] / det_scale, bboxes_list[o * 4 + 1] / det_scale,
                                    bboxes_list[o * 4 + 2] / det_scale, bboxes_list[o * 4 + 3] / det_scale});
                scores_in.push_back(scores_list[o]);
                for (int j = 0; j < 10; ++j) kpss_in[i * 10 + j] = kpss_list[o * 10 + j] / det_scale;
            }
            const auto keep = nms(boxes_in, scores_in, nms_thresh_);
            out.reserve(keep.size());
            for (int idx : keep) {
                InsightFaceBox bb;
                bb.bbox = boxes_in[idx];
                bb.score = scores_in[idx];
                for (int j = 0; j < 5; ++j) bb.kps.push_back({kpss_in[idx * 10 + j * 2], kpss_in[idx * 10 + j * 2 + 1]});
                out.push_back(std::move(bb));
            }
        }
        return true;
    }

} // namespace modeldeploy::vision::face

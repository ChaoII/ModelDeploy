#include "vision/tracking/mot/clear_metrics.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/iou_matching.h"

namespace modeldeploy::vision::tracking {
    namespace {
        constexpr float kIoUThreshold = 0.5f;

        float clamp01(float v) {
            if (v < 0.0f) return 0.0f;
            if (v > 1.0f) return 1.0f;
            return v;
        }
    }  // namespace

    Metrics compute_clear(const MotSequence& seq,
                          const std::vector<std::vector<TrackResult>>& predictions) {
        Metrics m;

        size_t n_frames = seq.frames.size();
        long long tp = 0, fp = 0, fn = 0, num_gt = 0, id_switches = 0;
        long long total_pred_frames = 0;
        // ID matching: every matched (track_id, gt_id) occurrence contributes weight 1.
        std::map<std::pair<int, int>, long long> id_w;
        std::map<int, int> track_gt;  // track_id -> last matched gt_id
        float sum_iou = 0.0f;
        long long num_matched = 0;

        for (size_t i = 0; i < n_frames; ++i) {
            const MotFrame& fr = seq.frames[i];
            const std::vector<TrackResult>& preds =
                (i < predictions.size()) ? predictions[i] : std::vector<TrackResult>{};

            const size_t n_g = fr.gt_boxes.size();
            const size_t n_p = preds.size();
            total_pred_frames += static_cast<long long>(n_p);
            num_gt += static_cast<long long>(n_g);

            // Greedy max-IoU matching: each prediction -> best still-unmatched GT with
            // IoU >= 0.5.
            std::vector<char> gt_used(n_g, 0);
            std::vector<int> pred_match(n_p, -1);
            for (size_t p = 0; p < n_p; ++p) {
                int best_g = -1;
                float best_iou = kIoUThreshold;
                for (size_t g = 0; g < n_g; ++g) {
                    if (gt_used[g]) continue;
                    float ov = iou(preds[p].box, fr.gt_boxes[g]);
                    if (ov >= best_iou) {
                        best_iou = ov;
                        best_g = static_cast<int>(g);
                    }
                }
                if (best_g >= 0) {
                    gt_used[best_g] = 1;
                    pred_match[p] = best_g;
                }
            }

            for (size_t p = 0; p < n_p; ++p) {
                int g = pred_match[p];
                if (g < 0) {  // prediction with no GT overlap >= 0.5 -> FP
                    ++fp;
                    continue;
                }
                ++tp;
                int track_id = preds[p].track_id;
                int gt_id = fr.gt_ids[static_cast<size_t>(g)];
                id_w[{track_id, gt_id}]++;
                float ov = iou(preds[p].box, fr.gt_boxes[static_cast<size_t>(g)]);
                sum_iou += ov;
                ++num_matched;

                auto it = track_gt.find(track_id);
                if (it != track_gt.end()) {
                    if (it->second != gt_id) ++id_switches;  // identity associated to a new GT
                    it->second = gt_id;
                } else {
                    track_gt[track_id] = gt_id;
                }
            }
            // 局限性 #1：#id_switches 按"预测 track"计数（该 track 首次切换到新 GT 时 +1），
            // 而非按 GT 计数（一个 GT 被多个 track 抢占时不会成比例计入）。
            fn += static_cast<long long>(n_g);
        }
        fn -= tp;  // unmatched GT boxes across all frames

        m.num_gt = static_cast<int>(num_gt);
        m.num_fp = static_cast<int>(fp);
        m.num_fn = static_cast<int>(fn);
        m.id_switches = static_cast<float>(id_switches);

        // MOTA = 1 - (FP + FN + IDS) / num_gt
        if (num_gt == 0) {
            m.mota = 1.0f;  // vacuous perfect
        } else {
            m.mota = clamp01(1.0f - static_cast<float>(fp + fn + id_switches) /
                                          static_cast<float>(num_gt));
        }

        // Precision / recall
        if (tp + fp > 0) m.precision = clamp01(static_cast<float>(tp) / static_cast<float>(tp + fp));
        if (num_gt > 0) m.recall = clamp01(static_cast<float>(tp) / static_cast<float>(num_gt));

        // IDF1: maximum-weight bipartite matching between predicted track ids and GT ids.
        // Weight = number of frames a (track_id, gt_id) pair was matched.
        std::map<int, int> pred_idx, gt_idx;
        for (const auto& kv : id_w) {
            if (!pred_idx.count(kv.first.first)) pred_idx[kv.first.first] = static_cast<int>(pred_idx.size());
            if (!gt_idx.count(kv.first.second)) gt_idx[kv.first.second] = static_cast<int>(gt_idx.size());
        }

        long long idtp = 0;
        if (!id_w.empty()) {
            const int np = static_cast<int>(pred_idx.size());
            const int ng = static_cast<int>(gt_idx.size());
            std::vector<std::vector<float>> wmat(static_cast<size_t>(np),
                                                 std::vector<float>(static_cast<size_t>(ng), 0.0f));
            for (const auto& kv : id_w)
                wmat[static_cast<size_t>(pred_idx[kv.first.first])][static_cast<size_t>(
                    gt_idx[kv.first.second])] = static_cast<float>(kv.second);

            float big = 0.0f;
            for (const auto& row : wmat)
                for (float v : row) big = std::max(big, v);
            big += 1.0f;  // +1 so weights are strictly > 0 after inversion

            std::vector<std::vector<float>> cost(static_cast<size_t>(np),
                                                 std::vector<float>(static_cast<size_t>(ng), 0.0f));
            for (int r = 0; r < np; ++r)
                for (int c = 0; c < ng; ++c)
                    cost[static_cast<size_t>(r)][static_cast<size_t>(c)] =
                        big - wmat[static_cast<size_t>(r)][static_cast<size_t>(c)];

            for (const auto& pr : linear_sum_assignment(cost)) {
                if (pr.first >= 0 && pr.second >= 0)
                    idtp += static_cast<long long>(
                        wmat[static_cast<size_t>(pr.first)][static_cast<size_t>(pr.second)]);
            }
        }

        const long long idfp = total_pred_frames - idtp;
        const long long idfn = num_gt - idtp;
        const long long iden = 2 * idtp + idfp + idfn;
        m.idf1 = (iden > 0) ? clamp01(static_cast<float>(2.0 * idtp) / static_cast<float>(iden))
                            : 0.0f;

        // HOTA (simplified): sqrt(det_acc * asso_acc)
        const long long den = tp + fp + fn;
        const float det_acc = (den > 0) ? static_cast<float>(tp) / static_cast<float>(den) : 1.0f;
        const float asso = (num_matched > 0) ? sum_iou / static_cast<float>(num_matched) : 1.0f;
        m.hota = clamp01(std::sqrt(det_acc * asso));

        return m;
    }
}

#include "vision/tracking/strongsort.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/iou_matching.h"

namespace modeldeploy::vision::tracking {
    // ---------------------------------------------------------------------------
    // Internal NSA (Noise Scale Adaptive) Kalman filter.
    //
    // StrongSORT's core distinction from a plain SORT Kalman filter is its
    // "noise scale adaptive" process/measurement model. We implement a faithful
    // but pragmatic, self-contained version of the same 8-dim constant-velocity
    // state used elsewhere ([cx, cy, a, h, vcx, vcy, va, vh]) with two adaptive
    // twists, so the shared KalmanFilter used by ByteTrack/BoT-SORT is untouched.
    //
    // Process-noise scaling (predict): the deeper reason is that a constant-
    // velocity model is more uncertain the faster the target moves — a high-
    // velocity target is far more likely to have changed direction/acceleration
    // than a stationary one. So before prediction we compute the current velocity
    // magnitude  v = sqrt(vcx^2 + vcy^2)  (pixels/frame) and scale the position-
    // related process-noise standard deviations by  pos_scale = 1 + k_vel * v.
    // Larger velocity => larger prediction uncertainty (bigger Q for the position
    // rows), which keeps the filter from "over-trusting" a stale constant-velocity
    // extrapolation. k_vel = 0.01 (so a 100 px/frame mover gets 2x position noise).
    //
    // Measurement-noise scaling (update): detection confidence tells us how much
    // to trust the observed box. High-confidence detections are weighted more
    // heavily (smaller R), low-confidence ones less (larger R). We scale the base
    // measurement noise by  r_scale = 1 / max(confidence, 0.05), clamped to
    // [0.5, 6.0], so R is not pathological for near-zero confidence.
    //
    // Both scalings are mild and hysteresis-free; with a stationary target
    // (v ~ 0) and a confident detection (conf ~ 1) this reduces to the standard
    // SORT update, so id-stability and non-divergence are preserved.
    // ---------------------------------------------------------------------------
    namespace {
        constexpr int kNSADim = 8;
        constexpr int kNSAMeas = 4;
        constexpr double kNSAStdWeightPosition = 1.0 / 20.0;
        constexpr double kNSAStdWeightVelocity = 1.0 / 160.0;
        constexpr double kNSAKVelScale = 0.01;

        struct NsaMat {
            int rows{0};
            int cols{0};
            std::vector<double> data;
            NsaMat() = default;
            NsaMat(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}
            double& operator()(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
            double operator()(int i, int j) const {
                return data[static_cast<size_t>(i) * cols + j];
            }
        };

        NsaMat nsa_diag(const std::vector<double>& d) {
            const int n = static_cast<int>(d.size());
            NsaMat m(n, n);
            for (int i = 0; i < n; ++i) m(i, i) = d[i];
            return m;
        }
        NsaMat nsa_mul(const NsaMat& a, const NsaMat& b) {
            NsaMat r(a.rows, b.cols);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < b.cols; ++j) {
                    double acc = 0.0;
                    for (int k = 0; k < a.cols; ++k) acc += a(i, k) * b(k, j);
                    r(i, j) = acc;
                }
            return r;
        }
        NsaMat nsa_t(const NsaMat& a) {
            NsaMat t(a.cols, a.rows);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) t(j, i) = a(i, j);
            return t;
        }
        NsaMat nsa_add(const NsaMat& a, const NsaMat& b) {
            NsaMat r(a.rows, a.cols);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) r(i, j) = a(i, j) + b(i, j);
            return r;
        }
        NsaMat nsa_sub(const NsaMat& a, const NsaMat& b) {
            NsaMat r(a.rows, a.cols);
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j) r(i, j) = a(i, j) - b(i, j);
            return r;
        }
        // Gauss-Jordan inverse with partial pivoting; empty on singular.
        NsaMat nsa_inv(const NsaMat& a) {
            const int n = a.rows;
            NsaMat aug(n, 2 * n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) aug(i, j) = a(i, j);
                aug(i, n + i) = 1.0;
            }
            for (int col = 0; col < n; ++col) {
                int piv = col;
                for (int r = col + 1; r < n; ++r)
                    if (std::abs(aug(r, col)) > std::abs(aug(piv, col))) piv = r;
                if (std::abs(aug(piv, col)) < 1e-12) return NsaMat();
                if (piv != col)
                    for (int j = 0; j < 2 * n; ++j) std::swap(aug(col, j), aug(piv, j));
                const double d = aug(col, col);
                for (int j = 0; j < 2 * n; ++j) aug(col, j) /= d;
                for (int r = 0; r < n; ++r) {
                    if (r == col) continue;
                    const double f = aug(r, col);
                    if (f == 0.0) continue;
                    for (int j = 0; j < 2 * n; ++j) aug(r, j) -= f * aug(col, j);
                }
            }
            NsaMat inv(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) inv(i, j) = aug(i, n + j);
            return inv;
        }

        std::array<double, 4> rect_to_xyah(const Rect2f& box) {
            const double cx = box.x + box.width * 0.5;
            const double cy = box.y + box.height * 0.5;
            const double h = box.height;
            const double a = (h > 0.0) ? box.width / h : 0.0;
            return {cx, cy, a, h};
        }
        Rect2f xyah_to_rect(const std::array<double, 4>& xyah) {
            const double h = xyah[3] > 0.0 ? xyah[3] : 1.0;
            const double w = xyah[2] * h;
            const double x = xyah[0] - w * 0.5;
            const double y = xyah[1] - h * 0.5;
            return {static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(w), static_cast<float>(h)};
        }
    } // namespace

    void StrongSortTracker::NSAKalman::init(const Rect2f& box) {
            const auto z = rect_to_xyah(box);
            for (int i = 0; i < kNSAMeas; ++i) mean_[i] = z[i];
            for (int i = kNSAMeas; i < kNSADim; ++i) mean_[i] = 0.0;
            const double h = z[3] > 0.0 ? z[3] : 1.0;
            const std::vector<double> std = {
                2.0 * kNSAStdWeightPosition * h, 2.0 * kNSAStdWeightPosition * h,
                1e-2,                             2.0 * kNSAStdWeightPosition * h,
                10.0 * kNSAStdWeightVelocity * h, 10.0 * kNSAStdWeightVelocity * h,
                1e-5,                             10.0 * kNSAStdWeightVelocity * h};
            for (auto& row : cov_) row.fill(0.0);
            for (int i = 0; i < kNSADim; ++i) cov_[i][i] = std[i] * std[i];
            initialized_ = true;
        }

        void StrongSortTracker::NSAKalman::predict() {
            if (!initialized_) return;
            const double h = mean_[3] > 0.0 ? mean_[3] : 1.0;
            // NSA process-noise scaling from the current velocity magnitude.
            const double vel = std::sqrt(mean_[4] * mean_[4] + mean_[5] * mean_[5]);
            const double pos_scale = 1.0 + kNSAKVelScale * vel;
            const std::vector<double> std_pos = {
                kNSAStdWeightPosition * h * pos_scale,
                kNSAStdWeightPosition * h * pos_scale,
                1e-2,
                kNSAStdWeightPosition * h * pos_scale};
            const std::vector<double> std_vel = {
                kNSAStdWeightVelocity * h, kNSAStdWeightVelocity * h,
                1e-5,                      kNSAStdWeightVelocity * h};
            std::vector<double> diag;
            diag.reserve(kNSADim);
            for (int i = 0; i < kNSAMeas; ++i) diag.push_back(std_pos[i] * std_pos[i]);
            for (int i = 0; i < kNSAMeas; ++i) diag.push_back(std_vel[i] * std_vel[i]);
            const NsaMat Q = nsa_diag(diag);

            NsaMat F(kNSADim, kNSADim);
            for (int i = 0; i < kNSADim; ++i) F(i, i) = 1.0;
            for (int i = 0; i < kNSAMeas; ++i) F(i, i + kNSAMeas) = 1.0;

            NsaMat mean(kNSADim, 1);
            for (int i = 0; i < kNSADim; ++i) mean(i, 0) = mean_[i];
            NsaMat cov(kNSADim, kNSADim);
            for (int i = 0; i < kNSADim; ++i)
                for (int j = 0; j < kNSADim; ++j) cov(i, j) = cov_[i][j];
            const NsaMat ft = nsa_t(F);

            NsaMat new_mean = nsa_mul(F, mean);
            NsaMat new_cov = nsa_add(nsa_mul(nsa_mul(F, cov), ft), Q);
            for (int i = 0; i < kNSADim; ++i) mean_[i] = new_mean(i, 0);
            for (int i = 0; i < kNSADim; ++i)
                for (int j = 0; j < kNSADim; ++j) cov_[i][j] = new_cov(i, j);
        }

        void StrongSortTracker::NSAKalman::update(const Rect2f& box, float confidence) {
            if (!initialized_) return;
            const auto xyah = rect_to_xyah(box);
            NsaMat z(kNSAMeas, 1);
            for (int i = 0; i < kNSAMeas; ++i) z(i, 0) = xyah[i];

            const double h = mean_[3] > 0.0 ? mean_[3] : 1.0;
            const double conf = std::max(static_cast<double>(confidence), 0.05);
            const double r_scale = std::clamp(1.0 / conf, 0.5, 6.0);
            const std::vector<double> std_pos = {
                kNSAStdWeightPosition * h * r_scale,
                kNSAStdWeightPosition * h * r_scale,
                1e-1 * r_scale,
                kNSAStdWeightPosition * h * r_scale};
            std::vector<double> diag;
            for (int i = 0; i < kNSAMeas; ++i) diag.push_back(std_pos[i] * std_pos[i]);
            const NsaMat R = nsa_diag(diag);

            NsaMat H(kNSAMeas, kNSADim);
            for (int i = 0; i < kNSAMeas; ++i) H(i, i) = 1.0;

            NsaMat mean(kNSADim, 1);
            for (int i = 0; i < kNSADim; ++i) mean(i, 0) = mean_[i];
            NsaMat cov(kNSADim, kNSADim);
            for (int i = 0; i < kNSADim; ++i)
                for (int j = 0; j < kNSADim; ++j) cov(i, j) = cov_[i][j];
            const NsaMat ht = nsa_t(H);

            const NsaMat s = nsa_add(nsa_mul(nsa_mul(H, cov), ht), R);
            const NsaMat sinv = nsa_inv(s);
            if (sinv.rows == 0) return;
            const NsaMat k = nsa_mul(nsa_mul(cov, ht), sinv);

            const NsaMat proj_mean = nsa_mul(H, mean);
            const NsaMat innovation = nsa_sub(z, proj_mean);
            const NsaMat k_innov = nsa_mul(k, innovation);

            NsaMat new_mean = nsa_add(mean, k_innov);
            const NsaMat kcov = nsa_mul(k, nsa_mul(s, nsa_t(k)));
            NsaMat new_cov = nsa_sub(cov, kcov);
            for (int i = 0; i < kNSADim; ++i) mean_[i] = new_mean(i, 0);
            for (int i = 0; i < kNSADim; ++i)
                for (int j = 0; j < kNSADim; ++j) cov_[i][j] = new_cov(i, j);
        }

        Rect2f StrongSortTracker::NSAKalman::get_state() const {
            if (!initialized_) return {};
            const std::array<double, 4> xyah = {mean_[0], mean_[1], mean_[2], mean_[3]};
            return xyah_to_rect(xyah);
        }

    StrongSortTracker::StrongSortTracker() { set_params(); }

    StrongSortTracker::~StrongSortTracker() = default;

    void StrongSortTracker::set_params(float track_thresh, float high_thresh, float low_thresh,
                                       int max_age, int min_hits, float iou_threshold,
                                       float match_thresh, float ema_alpha,
                                       float appearance_priority, bool with_cmc) {
        track_thresh_ = track_thresh;
        high_thresh_ = high_thresh;
        low_thresh_ = low_thresh;
        max_age_ = max_age;
        min_hits_ = min_hits;
        iou_threshold_ = iou_threshold;
        match_thresh_ = match_thresh;
        ema_alpha_ = ema_alpha;
        appearance_priority_ = appearance_priority;
        with_cmc_ = with_cmc;
        warp_ = cv::Mat::eye(2, 3, CV_32FC1);
    }

    void StrongSortTracker::set_reid(std::shared_ptr<ReidExtractor> reid) {
        reid_ = std::move(reid);
    }

    void StrongSortTracker::reset() {
        tracks_.clear();
        frame_counter_ = 0;
        next_id_ = 0;
        warp_valid_ = false;
        prev_gray_ = cv::Mat();
        prev_w_ = 0;
        prev_h_ = 0;
        have_prev_ = false;
        warp_ = cv::Mat::eye(2, 3, CV_32FC1);
    }

    std::vector<float> StrongSortTracker::l2_normalize(std::vector<float> v) {
        double sum = 0.0;
        for (const float x : v) sum += static_cast<double>(x) * static_cast<double>(x);
        if (sum <= 0.0) return v;
        const float norm = static_cast<float>(std::sqrt(sum));
        for (float& x : v) x /= norm;
        return v;
    }

    std::vector<float> StrongSortTracker::appearance_for(const Detection& d,
                                                         const ImageData* frame) const {
        if (!d.feature.empty()) {
            return l2_normalize(d.feature);
        }
        if (reid_ && reid_->is_initialized() && frame != nullptr && !frame->empty()) {
            const ImageData patch = frame->crop(d.box);
            if (!patch.empty()) {
                std::vector<float> feat = reid_->extract(patch);
                if (!feat.empty()) {
                    return l2_normalize(feat);
                }
            }
        }
        return {};
    }

    float StrongSortTracker::appearance_priority_cost(const Track& t,
                                                      const std::vector<float>& det_feat,
                                                      const Rect2f& track_box,
                                                      const Rect2f& det_box) const {
        const float iou_dist = std::max(0.0f, 1.0f - iou(track_box, det_box));
        if (t.ema_feature.empty() || det_feat.empty()) {
            return iou_dist; // no appearance -> pure IoU
        }
        const size_t n = std::min(t.ema_feature.size(), det_feat.size());
        double dot = 0.0;
        for (size_t i = 0; i < n; ++i) {
            dot += static_cast<double>(t.ema_feature[i]) * static_cast<double>(det_feat[i]);
        }
        // Both features are L2-normalized -> cosine distance = 1 - dot (clamped).
        float cos_dist = static_cast<float>(1.0 - dot);
        if (cos_dist < 0.0f) cos_dist = 0.0f;
        if (cos_dist > 2.0f) cos_dist = 2.0f;
        return appearance_priority_ * cos_dist +
               (1.0f - appearance_priority_) * iou_dist;
    }

    bool StrongSortTracker::estimate_camera_warp(const ImageData* frame) {
        warp_valid_ = false;
        if (!with_cmc_ || frame == nullptr || frame->empty()) {
            return false;
        }
        const int cw = frame->width();
        const int ch = frame->height();
        if (cw <= 0 || ch <= 0) {
            return false;
        }
        cv::Mat bgr;
        if (!frame->asMat(&bgr) || bgr.empty()) {
            return false;
        }
        cv::Mat cur_gray;
        if (bgr.channels() == 3) {
            cv::cvtColor(bgr, cur_gray, cv::COLOR_BGR2GRAY);
        } else if (bgr.channels() == 1) {
            cur_gray = bgr;
        } else {
            return false;
        }
        if (have_prev_ && prev_w_ == cw && prev_h_ == ch && !prev_gray_.empty()) {
            cv::Mat guess = prev_warp_;
            try {
                cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 1e-4);
                const cv::Mat cmd = cv::Mat::eye(2, 3, CV_32F);
                cv::Mat warp = guess.clone();
                cv::findTransformECC(prev_gray_, cur_gray, warp, cv::MOTION_AFFINE, criteria,
                                     cmd, 5);
                warp_ = warp;
                warp_valid_ = true;
            } catch (...) {
                warp_ = cv::Mat::eye(2, 3, CV_32FC1);
                warp_valid_ = false;
            }
        } else {
            warp_ = cv::Mat::eye(2, 3, CV_32FC1);
        }
        prev_warp_ = warp_.clone();
        prev_gray_ = cur_gray.clone();
        prev_w_ = cw;
        prev_h_ = ch;
        have_prev_ = true;
        return warp_valid_;
    }

    std::vector<TrackResult> StrongSortTracker::update(const std::vector<Detection>& detections,
                                                       const ImageData* frame, double timestamp) {
        (void)timestamp;
        ++frame_counter_;
        const int cur = frame_counter_;

        estimate_camera_warp(frame);

        auto candidate_box = [this](const Track& t) {
            Rect2f kf_box = t.kf.get_state();
            if (!(kf_box.width > 0.0f && kf_box.height > 0.0f)) {
                kf_box = t.box;
            }
            if (warp_valid_) {
                const auto W = [this](float x, float y) {
                    return cv::Point2f(warp_.at<float>(0, 0) * x + warp_.at<float>(0, 1) * y + warp_.at<float>(0, 2),
                                       warp_.at<float>(1, 0) * x + warp_.at<float>(1, 1) * y + warp_.at<float>(1, 2));
                };
                const cv::Point2f tl = W(kf_box.x, kf_box.y);
                const cv::Point2f tr = W(kf_box.x + kf_box.width, kf_box.y);
                const cv::Point2f bl = W(kf_box.x, kf_box.y + kf_box.height);
                const cv::Point2f br = W(kf_box.x + kf_box.width, kf_box.y + kf_box.height);
                const float x0 = std::min({tl.x, tr.x, bl.x, br.x});
                const float y0 = std::min({tl.y, tr.y, bl.y, br.y});
                const float x1 = std::max({tl.x, tr.x, bl.x, br.x});
                const float y1 = std::max({tl.y, tr.y, bl.y, br.y});
                kf_box.x = x0;
                kf_box.y = y0;
                kf_box.width = x1 - x0;
                kf_box.height = y1 - y0;
            }
            return kf_box;
        };

        for (auto& t : tracks_) {
            if (t.state != TrackState::Removed) {
                t.kf.predict();
            }
        }

        std::vector<const Detection*> high;
        std::vector<const Detection*> low;
        std::vector<std::vector<float>> high_feat;
        high.reserve(detections.size());
        low.reserve(detections.size());
        high_feat.reserve(detections.size());
        for (const auto& d : detections) {
            if (d.score >= track_thresh_) {
                high.push_back(&d);
                high_feat.push_back(appearance_for(d, frame));
            } else if (d.score >= low_thresh_) {
                low.push_back(&d);
            }
        }

        std::vector<int> pool;
        for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
            if (tracks_[static_cast<size_t>(i)].state != TrackState::Removed) {
                pool.push_back(i);
            }
        }

        std::vector<char> pool_used(pool.size(), 0);
        std::vector<char> high_used(high.size(), 0);

        // Stage 1 (+2): appearance-priority association of the whole pool with
        // high-confidence detections.
        if (!pool.empty() && !high.empty()) {
            std::vector<Rect2f> pb;
            std::vector<Rect2f> hb;
            pb.reserve(pool.size());
            hb.reserve(high.size());
            for (int pi : pool) {
                pb.push_back(candidate_box(tracks_[static_cast<size_t>(pi)]));
            }
            for (const auto* d : high) {
                hb.push_back(d->box);
            }

            std::vector<std::vector<float>> cost(pool.size(), std::vector<float>(high.size(), 0.0f));
            for (size_t r = 0; r < pool.size(); ++r) {
                const Track& t = tracks_[static_cast<size_t>(pool[r])];
                for (size_t c = 0; c < high.size(); ++c) {
                    cost[r][c] = appearance_priority_cost(t, high_feat[c], pb[r], hb[c]);
                }
            }

            for (const auto& pr : linear_sum_assignment(cost)) {
                const int r = pr.first;
                const int c = pr.second;
                if (r < 0 || c < 0) {
                    continue;
                }
                if (cost[static_cast<size_t>(r)][static_cast<size_t>(c)] < match_thresh_) {
                    pool_used[static_cast<size_t>(r)] = 1;
                    high_used[static_cast<size_t>(c)] = 1;
                    Track& t = tracks_[static_cast<size_t>(pool[static_cast<size_t>(r)])];
                    t.kf.update(high[static_cast<size_t>(c)]->box,
                                high[static_cast<size_t>(c)]->score);
                    t.state = TrackState::Tracked;
                    t.frame_id = cur;
                    t.time_since_update = 0;
                    ++t.hits;
                    t.score = high[static_cast<size_t>(c)]->score;
                    t.box = high[static_cast<size_t>(c)]->box;
                    t.label_id = high[static_cast<size_t>(c)]->label_id;
                    // EMA appearance update (StrongSORT). First association seeds it.
                    const std::vector<float>& det_feat = high_feat[static_cast<size_t>(c)];
                    if (!det_feat.empty()) {
                        if (t.ema_feature.empty()) {
                            t.ema_feature = det_feat;
                        } else {
                            const size_t n = std::min(t.ema_feature.size(), det_feat.size());
                            for (size_t i = 0; i < n; ++i) {
                                t.ema_feature[i] =
                                    ema_alpha_ * t.ema_feature[i] + (1.0f - ema_alpha_) * det_feat[i];
                            }
                            t.ema_feature = l2_normalize(t.ema_feature);
                        }
                    }
                }
            }
        }

        // Stage 3: re-associate unmatched, previously-tracked tracks with low dets.
        std::vector<char> low_used(low.size(), 0);
        if (!low.empty()) {
            std::vector<int> cand;
            std::vector<Rect2f> cb;
            for (size_t i = 0; i < pool.size(); ++i) {
                if (!pool_used[i] && tracks_[static_cast<size_t>(pool[i])].state == TrackState::Tracked) {
                    cand.push_back(static_cast<int>(i));
                    cb.push_back(candidate_box(tracks_[static_cast<size_t>(pool[i])]));
                }
            }
            if (!cand.empty()) {
                std::vector<Rect2f> lb;
                lb.reserve(low.size());
                for (const auto* d : low) {
                    lb.push_back(d->box);
                }
                const auto dist = iou_distance(cb, lb);
                for (const auto& pr : linear_sum_assignment(dist)) {
                    const int r = pr.first;
                    const int c = pr.second;
                    if (r < 0 || c < 0) {
                        continue;
                    }
                    if (dist[static_cast<size_t>(r)][static_cast<size_t>(c)] < match_thresh_) {
                        low_used[static_cast<size_t>(c)] = 1;
                        Track& t = tracks_[static_cast<size_t>(
                            pool[static_cast<size_t>(cand[static_cast<size_t>(r)])])];
                        t.kf.update(low[static_cast<size_t>(c)]->box,
                                    low[static_cast<size_t>(c)]->score);
                        t.state = TrackState::Lost; // low-confidence match -> uncertain
                        t.frame_id = cur;
                        t.time_since_update = 0;
                        t.score = low[static_cast<size_t>(c)]->score;
                        t.box = low[static_cast<size_t>(c)]->box;
                        t.label_id = low[static_cast<size_t>(c)]->label_id;
                    }
                }
            }
        }

        // New tracks from unmatched high-confidence detections.
        for (size_t c = 0; c < high.size(); ++c) {
            if (high_used[c]) {
                continue;
            }
            const Detection* d = high[c];
            Track t;
            t.kf.init(d->box);
            t.track_id = next_id_++;
            t.state = TrackState::Tracked;
            t.frame_id = cur;
            t.start_frame = cur;
            t.time_since_update = 0;
            t.hits = 1;
            t.score = d->score;
            t.box = d->box;
            t.label_id = d->label_id;
            t.ema_feature = high_feat[c]; // already normalized
            tracks_.push_back(std::move(t));
        }

        // Tracks not seen this frame -> mark lost; drop tracks lost beyond max_age.
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(), [&](Track& t) {
                if (t.state == TrackState::Removed) {
                    return true;
                }
                if (t.frame_id != cur && t.state == TrackState::Tracked) {
                    t.state = TrackState::Lost;
                }
                if (t.state == TrackState::Lost) {
                    t.time_since_update = cur - t.frame_id;
                    return (cur - t.frame_id) > max_age_;
                }
                return false;
            }),
            tracks_.end());

        // Output: every visibly-tracked track this frame, with its EMA feature.
        std::vector<TrackResult> out;
        out.reserve(tracks_.size());
        for (const auto& t : tracks_) {
            if (t.state != TrackState::Tracked) {
                continue;
            }
            TrackResult r;
            r.track_id = t.track_id;
            r.box = t.box;
            r.score = t.score;
            r.label_id = t.label_id;
            r.state = static_cast<int>(TrackState::Tracked);
            r.feature = t.ema_feature;
            out.push_back(r);
        }
        return out;
    }
} // namespace modeldeploy::vision::tracking

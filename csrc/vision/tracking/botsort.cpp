#include "vision/tracking/botsort.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/opencv.hpp>

#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/iou_matching.h"

namespace modeldeploy::vision::tracking {
    BotSortTracker::BotSortTracker() { set_params(); }

    BotSortTracker::~BotSortTracker() = default;

    void BotSortTracker::set_params(float track_thresh, float high_thresh, float low_thresh,
                                    int max_age, int min_hits, float iou_threshold,
                                    float match_thresh, float fuse_score_weight,
                                    float ema_alpha, bool with_cmc) {
        track_thresh_ = track_thresh;
        high_thresh_ = high_thresh;
        low_thresh_ = low_thresh;
        max_age_ = max_age;
        min_hits_ = min_hits;
        iou_threshold_ = iou_threshold;
        match_thresh_ = match_thresh;
        fuse_score_weight_ = fuse_score_weight;
        ema_alpha_ = ema_alpha;
        with_cmc_ = with_cmc;
        warp_ = cv::Mat::eye(2, 3, CV_32FC1);
    }

    void BotSortTracker::set_reid(std::shared_ptr<ReidExtractor> reid) {
        reid_ = std::move(reid);
    }

    void BotSortTracker::reset() {
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

    bool BotSortTracker::normalize(std::vector<float>& v) const {
        double sum = 0.0;
        for (const float x : v) {
            sum += static_cast<double>(x) * static_cast<double>(x);
        }
        if (sum <= 0.0) {
            return false;
        }
        const float norm = static_cast<float>(std::sqrt(sum));
        for (float& x : v) {
            x /= norm;
        }
        return true;
    }

    std::vector<float> BotSortTracker::appearance_for(const Detection& d,
                                                      const ImageData* frame) const {
        // Preference 1: the detection already carries an inline feature (the test
        // path and/or an upstream ReID stage).
        if (!d.feature.empty()) {
            std::vector<float> feat = d.feature;
            normalize(feat);
            return feat;
        }
        // Preference 2: extract from the frame crop with an attached extractor.
        if (reid_ && reid_->is_initialized() && frame != nullptr && !frame->empty()) {
            const ImageData patch = frame->crop(d.box);
            if (!patch.empty()) {
                std::vector<float> feat = reid_->extract(patch);
                if (!feat.empty()) {
                    return feat;
                }
            }
        }
        return {};
    }

    float BotSortTracker::fused_cost(const Track& t, const std::vector<float>& det_feat,
                                     const Rect2f& track_box,
                                     const Rect2f& det_box) const {
        float iou_dist = 1.0f - iou(track_box, det_box);
        if (iou_dist < 0.0f) {
            iou_dist = 0.0f;
        }
        if (t.ema_feature.empty() || det_feat.empty()) {
            return iou_dist; // no appearance -> pure IoU
        }
        const size_t n = std::min(t.ema_feature.size(), det_feat.size());
        double dot = 0.0;
        for (size_t i = 0; i < n; ++i) {
            dot += static_cast<double>(t.ema_feature[i]) * static_cast<double>(det_feat[i]);
        }
        // Both features are L2-normalized -> cosine distance = 1 - dot (clamped to
        // [0, 2] for safety against tiny numeric drift).
        float cos_dist = static_cast<float>(1.0 - dot);
        if (cos_dist < 0.0f) {
            cos_dist = 0.0f;
        }
        if (cos_dist > 2.0f) {
            cos_dist = 2.0f;
        }
        return fuse_score_weight_ * iou_dist + (1.0f - fuse_score_weight_) * cos_dist;
    }

    bool BotSortTracker::estimate_camera_warp(const ImageData* frame) {
        // CMC requires a real frame; with frame == nullptr (the common test path)
        // it is a strict no-op (identity warp).
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
            // Estimate the affine transform mapping the previous frame into the
            // current one (accumulate on the prior warp as the initial guess).
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
                // findTransformECC can throw on degenerate/homogeneous or tiny frames;
                // fall back to identity (no compensation) — never crash.
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

    std::vector<TrackResult> BotSortTracker::update(const std::vector<Detection>& detections,
                                                    const ImageData* frame, double timestamp) {
        (void)timestamp;
        ++frame_counter_;
        const int cur = frame_counter_;

        // Camera motion compensation before prediction/matching.
        estimate_camera_warp(frame);

        auto candidate_box = [this](const Track& t) {
            Rect2f kf_box = t.kf.get_state();
            if (!(kf_box.width > 0.0f && kf_box.height > 0.0f)) {
                kf_box = t.box;
            }
            if (warp_valid_) {
                // Warp the predicted box's corners into the current frame via the
                // estimated affine transform, keeping the axis-aligned bounding box.
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

        // Step 1: predict the motion of every live track.
        for (auto& t : tracks_) {
            if (t.state != TrackState::Removed) {
                t.kf.predict();
            }
        }

        // Step 2: split detections into high/low confidence and cache appearances.
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

        // Candidate track pool: every live track (tracked or recently lost).
        std::vector<int> pool;
        for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
            if (tracks_[static_cast<size_t>(i)].state != TrackState::Removed) {
                pool.push_back(i);
            }
        }

        std::vector<char> pool_used(pool.size(), 0);
        std::vector<char> high_used(high.size(), 0);

        // Stage 1 (+2): associate the whole pool with high-confidence detections.
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
                    cost[r][c] =
                        fused_cost(t, high_feat[c], pb[r], hb[c]);
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
                    t.kf.update(high[static_cast<size_t>(c)]->box);
                    t.state = TrackState::Tracked;
                    t.frame_id = cur;
                    t.time_since_update = 0;
                    ++t.hits;
                    t.score = high[static_cast<size_t>(c)]->score;
                    t.box = high[static_cast<size_t>(c)]->box;
                    t.label_id = high[static_cast<size_t>(c)]->label_id;
                    // EMA appearance update (BoT-SORT). First association seeds the EMA.
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
                            normalize(t.ema_feature);
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
                        Track& t = tracks_[static_cast<size_t>(pool[static_cast<size_t>(cand[static_cast<size_t>(r)])])];
                        t.kf.update(low[static_cast<size_t>(c)]->box);
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

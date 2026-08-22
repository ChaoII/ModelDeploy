#include "vision/tracking/bytetrack.h"

#include <algorithm>
#include <utility>

#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/iou_matching.h"

namespace modeldeploy::vision::tracking {
    ByteTracker::ByteTracker() { set_params(); }

    ByteTracker::~ByteTracker() = default;

    void ByteTracker::set_params(float track_thresh, float high_thresh, float low_thresh,
                                 int max_age, int min_hits, float iou_threshold) {
        track_thresh_ = track_thresh;
        high_thresh_ = high_thresh;
        low_thresh_ = low_thresh;
        max_age_ = max_age;
        min_hits_ = min_hits;
        iou_threshold_ = iou_threshold;
        // Canonical ByteTrack match_thresh: distance gate, default 0.8 (~IoU 0.2).
        match_thresh_ = 0.8f;
    }

    void ByteTracker::reset() {
        tracks_.clear();
        frame_counter_ = 0;
        next_id_ = 0;
    }

    std::unique_ptr<BaseTracker> ByteTracker::clone() const {
        return std::make_unique<ByteTracker>(*this);
    }

    std::vector<TrackResult> ByteTracker::update(const std::vector<Detection>& detections,
                                                 const ImageData* frame, double timestamp) {
        (void)frame;
        (void)timestamp;
        ++frame_counter_;
        const int cur = frame_counter_;

        // Step 1: predict the motion of every live track.
        for (auto& t : tracks_) {
            if (t.state != TrackState::Removed) t.kf.predict();
        }

        // Step 2: split detections into high and low confidence groups.
        std::vector<const Detection*> high;
        std::vector<const Detection*> low;
        high.reserve(detections.size());
        low.reserve(detections.size());
        for (const auto& d : detections) {
            if (d.score >= track_thresh_) {
                high.push_back(&d);
            } else if (d.score >= low_thresh_) {
                low.push_back(&d);
            }
        }

        // Candidate track pool: every live track (tracked or recently lost).
        std::vector<int> pool;
        for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
            if (tracks_[static_cast<size_t>(i)].state != TrackState::Removed) pool.push_back(i);
        }

        auto candidate_box = [](const Track& t) {
            const Rect2f kf_box = t.kf.get_state();
            if (kf_box.width > 0.0f && kf_box.height > 0.0f) return kf_box;
            return t.box;
        };

        std::vector<char> pool_used(pool.size(), 0);
        std::vector<char> high_used(high.size(), 0);

        // Stage 1 (+2): associate the whole pool with high-confidence detections.
        if (!pool.empty() && !high.empty()) {
            std::vector<Rect2f> pb;
            std::vector<Rect2f> hb;
            pb.reserve(pool.size());
            hb.reserve(high.size());
            for (int pi : pool) pb.push_back(candidate_box(tracks_[static_cast<size_t>(pi)]));
            for (const auto* d : high) hb.push_back(d->box);

            const auto dist = iou_distance(pb, hb);
            for (const auto& pr : linear_sum_assignment(dist)) {
                const int r = pr.first;
                const int c = pr.second;
                if (r < 0 || c < 0) continue;
                if (dist[static_cast<size_t>(r)][static_cast<size_t>(c)] < match_thresh_) {
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
                for (const auto* d : low) lb.push_back(d->box);

                const auto dist = iou_distance(cb, lb);
                for (const auto& pr : linear_sum_assignment(dist)) {
                    const int r = pr.first;
                    const int c = pr.second;
                    if (r < 0 || c < 0) continue;
                    if (dist[static_cast<size_t>(r)][static_cast<size_t>(c)] < match_thresh_) {
                        low_used[static_cast<size_t>(c)] = 1;
                        Track& t = tracks_[static_cast<size_t>(pool[static_cast<size_t>(cand[static_cast<size_t>(r)])])];
                        t.kf.update(low[static_cast<size_t>(c)]->box);
                        t.state = TrackState::Lost;  // low-confidence match -> uncertain
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
            if (high_used[c]) continue;
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
            tracks_.push_back(std::move(t));
        }

        // Tracks not seen this frame -> mark lost; drop tracks lost beyond max_age.
        // A track seen this frame keeps frame_id == cur and stays active/visible.
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(), [&](Track& t) {
                if (t.state == TrackState::Removed) return true;
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

        // Output: every visibly-tracked (not removed, not lost) track this frame.
        std::vector<TrackResult> out;
        out.reserve(tracks_.size());
        for (const auto& t : tracks_) {
            if (t.state != TrackState::Tracked) continue;
            TrackResult r;
            r.track_id = t.track_id;
            r.box = t.box;
            r.score = t.score;
            r.label_id = t.label_id;
            r.state = static_cast<int>(TrackState::Tracked);
            r.feature.clear();
            out.push_back(r);
        }
        return out;
    }
}

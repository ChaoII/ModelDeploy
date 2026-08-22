#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/matching/kalman_filter.h"

namespace modeldeploy::vision::tracking {
    // ByteTrack-style two-stage matching tracker (Yifu Zhang, 2021).
    // Associates high-confidence detections with tracks first, then re-associates
    // leftover tracks with low-confidence detections for occlusion robustness.
    class MODELDEPLOY_CXX_EXPORT ByteTracker : public BaseTracker {
    public:
        ByteTracker();
        ~ByteTracker() override;

        void set_params(float track_thresh = 0.5f, float high_thresh = 0.5f,
                        float low_thresh = 0.1f, int max_age = 30,
                        int min_hits = 3, float iou_threshold = 0.3f);

        std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                        const ImageData* frame = nullptr,
                                        double timestamp = -1) override;
        void reset() override;

    private:
        struct Track {
            KalmanFilter kf;
            int track_id{-1};
            TrackState state{TrackState::New};
            int frame_id{0};
            int start_frame{0};
            int time_since_update{0};
            int hits{0};
            float score{0.0f};
            Rect2f box{};
            int label_id{0};
        };

        std::vector<Track> tracks_;
        int frame_counter_{0};
        int next_id_{0};
        float track_thresh_{0.5f};
        float high_thresh_{0.5f};
        float low_thresh_{0.1f};
        int max_age_{30};
        int min_hits_{3};
        // Active association gate: a matched pair is accepted only when its IoU
        // distance (1 - IoU) is below match_thresh_ (default 0.8 => min required
        // IoU of 0.2), matching canonical ByteTrack's match_thresh semantics.
        float match_thresh_{0.8f};
        // Kept in the public set_params signature for API compatibility. The
        // effective gating threshold is match_thresh_ (above); this value is
        // stored but is not consulted by the association stages.
        float iou_threshold_{0.3f};
    };
}

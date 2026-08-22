#pragma once
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/matching/kalman_filter.h"
#include "vision/tracking/reid_extractor.h"

namespace modeldeploy::vision::tracking {
    // BoT-SORT tracker (Aharon et al., 2022): ByteTrack-style two-stage matching
    // fused with ReID appearance. Each track keeps an EMA-smoothed appearance
    // embedding; the association cost is the weighted sum of IoU distance and
    // cosine distance between the track's EMA feature and the detection feature.
    // Optional Camera Motion Compensation (ECC) warps predicted boxes into the
    // current frame before matching. CMC requires real frames (grayscale) and is
    // a no-op (identity) when update() is called with frame == nullptr, so the
    // tracker degrades gracefully to pure IoU+appearance matching.
    class MODELDEPLOY_CXX_EXPORT BotSortTracker : public BaseTracker {
    public:
        BotSortTracker();
        ~BotSortTracker() override;

        void set_params(float track_thresh = 0.5f, float high_thresh = 0.5f,
                        float low_thresh = 0.1f, int max_age = 30,
                        int min_hits = 3, float iou_threshold = 0.3f,
                        float match_thresh = 0.8f, float fuse_score_weight = 0.5f,
                        float ema_alpha = 0.9f, bool with_cmc = true);
        void set_reid(std::shared_ptr<ReidExtractor> reid);

        std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                        const ImageData* frame = nullptr,
                                        double timestamp = -1) override;
        void reset() override;
        std::unique_ptr<BaseTracker> clone() const override;

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
            std::vector<float> ema_feature;
        };

        // Returns the normalized appearance feature for a detection, honoring the
        // priority: inline Detection.feature > ReidExtractor on a frame crop.
        // Empty vector means "no appearance" (fall back to pure IoU).
        std::vector<float> appearance_for(const Detection& d, const ImageData* frame) const;

        // L2-normalizes a vector in place (returns false if zero-norm).
        bool normalize(std::vector<float>& v) const;

        // Fused cost: fuse_weight * (1 - IoU) + (1 - fuse_weight) * cosine_dist.
        // Falls back to pure IoU cost when either feature is empty.
        float fused_cost(const Track& t, const std::vector<float>& det_feat,
                         const Rect2f& track_box, const Rect2f& det_box) const;

        // Camera motion compensation state.
        bool estimate_camera_warp(const ImageData* frame);

        std::vector<Track> tracks_;
        int frame_counter_{0};
        int next_id_{0};
        float track_thresh_{0.5f};
        float high_thresh_{0.5f};
        float low_thresh_{0.1f};
        int max_age_{30};
        int min_hits_{3};
        float iou_threshold_{0.3f};
        float match_thresh_{0.8f};
        float fuse_score_weight_{0.5f};
        float ema_alpha_{0.9f};
        bool with_cmc_{true};

        std::shared_ptr<ReidExtractor> reid_;

        // CMC bookkeeping.
        bool warp_valid_{false};
        cv::Mat warp_{2, 3, CV_32FC1};
        cv::Mat prev_warp_{2, 3, CV_32FC1};
        cv::Mat prev_gray_;
        int prev_w_{0};
        int prev_h_{0};
        bool have_prev_{false};
    };
} // namespace modeldeploy::vision::tracking

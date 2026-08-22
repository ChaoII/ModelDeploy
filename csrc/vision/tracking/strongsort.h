#pragma once
#include <array>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/reid_extractor.h"

namespace modeldeploy::vision::tracking {
    // StrongSORT tracker (Du et al., 2023): a DeepSORT+ lineage tracker that
    // improves on SORT/ByteTrack with an NSA (Noise Scale Adaptive) Kalman filter
    // and appearance-priority matching. Each track keeps an EMA-smoothed deep
    // embedding; association weights the cosine distance between the track's EMA
    // feature and the detection's feature more heavily than the IoU distance
    // (appearance-priority, default weight 0.7 > BoT-SORT's 0.5). Optional
    // Camera Motion Compensation (ECC) warps predicted boxes into the current
    // frame before matching; it is a strict no-op (identity) when update() is
    // called with frame == nullptr, so the tracker degrades gracefully.
    class MODELDEPLOY_CXX_EXPORT StrongSortTracker : public BaseTracker {
    public:
        StrongSortTracker();
        ~StrongSortTracker() override;

        void set_params(float track_thresh = 0.5f, float high_thresh = 0.5f,
                        float low_thresh = 0.1f, int max_age = 30,
                        int min_hits = 3, float iou_threshold = 0.3f,
                        float match_thresh = 0.8f, float ema_alpha = 0.9f,
                        float appearance_priority = 0.7f, bool with_cmc = true);
        void set_reid(std::shared_ptr<ReidExtractor> reid);

        std::vector<TrackResult> update(const std::vector<Detection>& detections,
                                        const ImageData* frame = nullptr,
                                        double timestamp = -1) override;
        void reset() override;

    private:
        // Internal NSA (Noise Scale Adaptive) Kalman filter — StrongSORT's
        // adaptive-uncertainty variant of the 8-dim constant-velocity model. It
        // holds only flat arrays so the filter is a complete type usable by value
        // inside Track; the matrix algebra lives in strongsort.cpp.
        struct NSAKalman {
            NSAKalman() = default;
            void init(const Rect2f& box);
            void predict();
            void update(const Rect2f& box, float confidence);
            Rect2f get_state() const;
            std::array<double, 8> mean_{};
            std::array<std::array<double, 8>, 8> cov_{};
            bool initialized_{false};
        };

        struct Track {
            NSAKalman kf;
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

        static std::vector<float> l2_normalize(std::vector<float> v);

        // Returns the normalized appearance feature for a detection, honoring the
        // priority: inline Detection.feature > ReidExtractor on a frame crop.
        // Empty vector means "no appearance" (fall back to pure IoU).
        std::vector<float> appearance_for(const Detection& d, const ImageData* frame) const;

        // Appearance-priority cost: appearance_priority * cosine_dist +
        // (1 - appearance_priority) * (1 - IoU) when both features are non-empty;
        // falls back to pure IoU distance otherwise.
        float appearance_priority_cost(const Track& t, const std::vector<float>& det_feat,
                                       const Rect2f& track_box, const Rect2f& det_box) const;

        // Camera motion compensation (ECC), crash-safe; no-op without a real frame.
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
        float ema_alpha_{0.9f};
        float appearance_priority_{0.7f};
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

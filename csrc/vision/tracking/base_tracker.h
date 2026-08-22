#pragma once
#include <memory>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::tracking {
    struct MODELDEPLOY_CXX_EXPORT Detection {
        Rect2f box; float score{}; int label_id{};
        std::vector<float> feature;
    };
    struct MODELDEPLOY_CXX_EXPORT TrackResult {
        int track_id{-1}; Rect2f box; float score{}; int label_id{};
        int state{0}; std::vector<float> feature;
    };
    enum class TrackState : int { New = 0, Tracked = 1, Lost = 2, Removed = 3 };
    class MODELDEPLOY_CXX_EXPORT BaseTracker {
    public:
        virtual std::vector<TrackResult> update(
            const std::vector<Detection>& detections,
            const ImageData* frame = nullptr, double timestamp = -1) = 0;
        virtual void reset() = 0;
        // Deep(-enough) clone of the tracker state. Used by the capi for the
        // non-mutating capacity query (md_tracker_capacity) so callers can size
        // their output buffer before the single stateful update() commit, without
        // advancing the real tracker's frame counter / Kalman state. This is a
        // query/commit *mechanism* only; it does not change tracking semantics.
        virtual std::unique_ptr<BaseTracker> clone() const = 0;
        virtual ~BaseTracker() = default;
    };
    inline std::vector<TrackResult> empty_update() { return {}; }
}

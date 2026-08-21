#pragma once
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
        virtual ~BaseTracker() = default;
    };
    inline std::vector<TrackResult> empty_update() { return {}; }
}

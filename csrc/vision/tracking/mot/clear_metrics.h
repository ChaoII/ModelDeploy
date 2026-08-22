#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/mot/mot_loader.h"

namespace modeldeploy::vision::tracking {
    // CLEAR / IDF1 / HOTA benchmark metrics for a tracking run.
    //
    // HOTA (simplified) formula used here:
    //     det  = TP / (TP + FP + FN)                 detection accuracy in [0,1]
    //     asso = mean IoU over all matched (pred, GT) pairs   association accuracy in [0,1]
    //     hota = sqrt(det * asso)
    // This is a simplified HOTA: det measures how completely/correctly objects were
    // found, asso measures how tightly the associated boxes overlap their GT. A perfect
    // tracker (all GT found, no FP/FN, boxes identical to GT, no ID switches) gives
    // det = asso = 1 => hota = 1. It is not the full official HOTA (which uses a global
    // alignment of identities); it substitutes the summary association-independent mean IoU.
    struct MODELDEPLOY_CXX_EXPORT Metrics {
        float mota = 0.f;       // MOTA in [0,1], 1 = perfect
        float idf1 = 0.f;       // IDF1 in [0,1]
        float hota = 0.f;       // simplified HOTA (see formula above) in [0,1]
        float id_switches = 0.f;
        float precision = 0.f;  // TP / (TP + FP)
        float recall = 0.f;     // TP / num_gt
        int num_fp = 0;
        int num_fn = 0;
        int num_gt = 0;
    };

    // predictions[i] must correspond to seq.frames[i] (per-frame TrackResult lists).
    MODELDEPLOY_CXX_EXPORT Metrics compute_clear(
        const MotSequence& seq, const std::vector<std::vector<TrackResult>>& predictions);
}

#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy::vision::tracking {
    // One video frame of a MOTChallenge-style sequence.
    struct MODELDEPLOY_CXX_EXPORT MotFrame {
        int frame_id = 0;
        std::vector<Rect2f> gt_boxes;  // ground-truth boxes this frame (with GT ids)
        std::vector<int> gt_ids;       // parallel to gt_boxes, GT object ids
        std::vector<Detection> dets;   // detector outputs (obj_id=-1, conf=score) to feed trackers
    };

    struct MODELDEPLOY_CXX_EXPORT MotSequence {
        std::vector<MotFrame> frames;  // sorted by ascending frame_id
        int num_gt_ids = 0;            // number of distinct GT object ids
    };

    // Loads a MOTChallenge-style file. Rows have columns:
    //   frame_id, obj_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    //   - GT rows: conf == -1 (obj_id = real id) -> gt_boxes / gt_ids.
    //   - Detection rows: conf >= 0 -> dets (Detection{box, score = conf column, label 0}).
    // Lines with < 7 columns, non-numeric rows, empty/comment lines are skipped.
    // class / visibility columns are ignored.
    MODELDEPLOY_CXX_EXPORT MotSequence load_mot_sequence(const std::string& filepath);
    MODELDEPLOY_CXX_EXPORT MotSequence load_mot_sequence_txt(const std::string& filepath);
}

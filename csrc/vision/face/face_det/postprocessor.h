//
// Created by aichao on 2025/2/20.
//

#pragma once

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"


namespace modeldeploy::vision::face {
    class MODELDEPLOY_CXX_EXPORT ScrfdPostprocessor {
    public:
        ScrfdPostprocessor();
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<std::vector<DetectionLandmarkResult>>* results,
                 const std::vector<LetterBoxRecord>& letter_box_records);

        /// Set conf_threshold, default 0.25
        void set_conf_threshold(const float& conf_threshold) {
            conf_threshold_ = conf_threshold;
        }

        /// Get conf_threshold, default 0.25
        [[nodiscard]] float get_conf_threshold() const { return conf_threshold_; }

        /// Set nms_threshold, default 0.5
        void set_nms_threshold(const float& nms_threshold) {
            nms_threshold_ = nms_threshold;
        }

        /// Get nms_threshold, default 0.5
        [[nodiscard]] float get_nms_threshold() const { return nms_threshold_; }


        /// Set landmarks_per_face, default 5
        void set_landmarks_per_face(const int& landmarks_per_face) {
            landmarks_per_face_ = landmarks_per_face;
        }

        /// Get landmarks_per_face, default 5
        [[nodiscard]] int get_landmarks_per_face() const { return landmarks_per_face_; }

    protected:
        void generate_points(int width, int height);
        float conf_threshold_;
        float nms_threshold_;
        bool is_dynamic_input_ = false;
        bool center_points_is_update_ = false;

        typedef struct {
            float cx;
            float cy;
        } SCRFDPoint;

        /// Argument for image postprocessing step, downsample strides (namely, steps) for SCRFD to generate anchors, will take (8,16,32) as default values
        std::vector<int> downsample_strides_{8, 16, 32};
        /// Argument for image postprocessing step, anchor number of each stride, default 2
        unsigned int num_anchors_ = 2;

        /// Argument for image postprocessing step, the upperbound number of boxes processed by nms, default 30000
        int max_nms_ = 30000;
        /// Argument for image postprocessing step, the outputs of onnx file with key points features or not, default true
        bool use_kps_ = true;
        /// Argument for image postprocessing step, landmarks_per_face, default 5 in SCRFD
        int landmarks_per_face_ = 5;
        std::unordered_map<int, std::vector<SCRFDPoint>> center_points_;
    };
} // namespace modeldeploy

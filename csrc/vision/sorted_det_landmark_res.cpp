//
// Created by aichao on 2025/3/25.
//

#include "csrc/vision/utils.h"

namespace modeldeploy::vision::utils {
    void sort_detection_result(DetectionLandmarkResult* result) {
        // sort face detection results with landmarks or not.
        if (result->boxes.empty()) {
            return;
        }
        int landmarks_per_instance = result->landmarks_per_instance;
        if (landmarks_per_instance > 0) {
            if (
                result->landmarks.size() != result->boxes.size() * landmarks_per_instance) {
                std::cerr << "The size of landmarks != boxes.size * landmarks_per_face." << std::endl;
            }
        }

        // argsort for scores.
        std::vector<size_t> indices;
        indices.resize(result->boxes.size());
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            indices[i] = i;
        }
        std::vector<float>& scores = result->scores;
        std::sort(indices.begin(), indices.end(),
                  [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

        // reorder boxes, scores, landmarks (if have).
        DetectionLandmarkResult backup(*result);
        result->clear();
        // don't forget to reset the landmarks_per_face
        // before apply Reserve method.
        result->landmarks_per_instance = landmarks_per_instance;
        result->reserve(indices.size());
        if (landmarks_per_instance > 0) {
            for (const auto& indice : indices) {
                result->boxes.emplace_back(backup.boxes[indice]);
                result->scores.push_back(backup.scores[indice]);
                result->label_ids.push_back(backup.label_ids[indice]);
                for (size_t j = 0; j < landmarks_per_instance; ++j) {
                    result->landmarks.emplace_back(
                        backup.landmarks[indice * landmarks_per_instance + j]);
                }
            }
        }
        else {
            for (const auto& indice : indices) {
                result->boxes.emplace_back(backup.boxes[indice]);
                result->scores.push_back(backup.scores[indice]);
                result->label_ids.push_back(backup.label_ids[indice]);
            }
        }
    }
}

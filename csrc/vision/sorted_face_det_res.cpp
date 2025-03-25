//
// Created by aichao on 2025/3/25.
//

#include "csrc/vision/utils.h"
namespace modeldeploy::vision::utils {
void sort_detection_result(FaceDetectionResult* result) {
    // sort face detection results with landmarks or not.
    if (result->boxes.size() == 0) {
        return;
    }
    int landmarks_per_face = result->landmarks_per_face;
    if (landmarks_per_face > 0) {
        if(
            (result->landmarks.size() != result->boxes.size() * landmarks_per_face)){
            std::cerr<<"The size of landmarks != boxes.size * landmarks_per_face."<<std::endl;
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
    FaceDetectionResult backup(*result);
    result->Clear();
    // don't forget to reset the landmarks_per_face
    // before apply Reserve method.
    result->landmarks_per_face = landmarks_per_face;
    result->Reserve(indices.size());
    if (landmarks_per_face > 0) {
        for (size_t i = 0; i < indices.size(); ++i) {
            result->boxes.emplace_back(backup.boxes[indices[i]]);
            result->scores.push_back(backup.scores[indices[i]]);
            for (size_t j = 0; j < landmarks_per_face; ++j) {
                result->landmarks.emplace_back(
                    backup.landmarks[indices[i] * landmarks_per_face + j]);
            }
        }
    } else {
        for (size_t i = 0; i < indices.size(); ++i) {
            result->boxes.emplace_back(backup.boxes[indices[i]]);
            result->scores.push_back(backup.scores[indices[i]]);
        }
    }
}}
//
// Created by AC on 2024-12-25.
//

#include "qs_clarity.h"

namespace seeta {

    QualityOfClarityEx::QualityOfClarityEx(const std::string &model_dir) {
        m_lbn = std::make_shared<QualityOfLBN>(ModelSetting(model_dir + "/quality_lbn.csta"));
        m_marker = std::make_shared<FaceLandmarker>(ModelSetting(model_dir + "/face_landmarker_pts68.csta"));
    }

    QualityResult QualityOfClarityEx::check(const SeetaImageData &image, const SeetaRect &face,
                                            const SeetaPointF *points, int32_t N) {
        // assert(N == 68);
        auto points68 = m_marker->mark(image, face);
        int light, blur, noise;
        m_lbn->Detect(image, points68.data(), &light, &blur, &noise);
        if (blur == QualityOfLBN::BLUR) {
            return {QualityLevel::LOW, 0};
        } else {
            return {QualityLevel::HIGH, 1};
        }
    }

    void QualityOfClarityEx::set_blur_threshold(float blur_thresh) {
        m_lbn->set(QualityOfLBN::PROPERTY_BLUR_THRESH, blur_thresh);
    }
}
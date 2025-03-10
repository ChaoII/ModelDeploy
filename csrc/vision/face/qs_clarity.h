//
// Created by AC on 2024-12-25.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/QualityStructure.h"
#include "seeta/QualityOfLBN.h"

namespace seeta {
    class MODELDEPLOY_CXX_EXPORT QualityOfClarityEx : public QualityRule {
    public:
        explicit QualityOfClarityEx(const std::string& model_dir);

        void set_blur_threshold(float blur_thresh) const;

        QualityResult
        check(const SeetaImageData& image, const SeetaRect& face, const SeetaPointF* points, int32_t N) override;

    private:
        std::shared_ptr<QualityOfLBN> m_lbn;
        std::shared_ptr<FaceLandmarker> m_marker;
    };
}

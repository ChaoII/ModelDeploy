//
// Created by AC on 2024-12-25.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/QualityStructure.h"

namespace seeta {
    class MODELDEPLOY_CXX_EXPORT QualityOfNoMask : public QualityRule {
    public:
        explicit QualityOfNoMask(std::shared_ptr<seeta::FaceLandmarker> ld_);

        QualityResult check(const SeetaImageData& image, const SeetaRect& face,
                            const SeetaPointF* points, int32_t N) override;

    private:
        std::shared_ptr<seeta::FaceLandmarker> m_marker;
    };
}

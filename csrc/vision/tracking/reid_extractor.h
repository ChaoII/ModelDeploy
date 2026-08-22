//
// Created for MOT tracking ReID appearance extraction.
//

#pragma once

#include <string>
#include <vector>
#include "core/md_decl.h"
#include "base_model.h"
#include "runtime/runtime.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::tracking {
    // ReID appearance extractor. Runs an OSNet/ResNet ONNX model (via ORT) on a
    // cropped person patch (256x128, ImageNet-normalized, CHW) and returns the
    // L2-normalized embedding.
    class MODELDEPLOY_CXX_EXPORT ReidExtractor : public BaseModel {
    public:
        ReidExtractor() = default;

        // Loads the ONNX model. Returns false on empty path or init failure.
        bool init(const std::string& onnx, const RuntimeOption& opt);

        // Extracts appearance embedding from an already-cropped patch.
        // Returns empty vector if not initialized, on an empty patch, or on
        // inference/preprocess failure.
        std::vector<float> extract(const ImageData& patch);

        bool is_initialized() const override;

        // Divides each element by the vector's L2 norm. Returns the vector
        // unchanged when the norm is zero (avoids producing NaN).
        std::vector<float> l2_normalize(const std::vector<float>& v) const;

        // Sets the model input spatial size used for resize (default 256x128).
        void set_input_size(int h, int w);

    private:
        int input_h_ = 256;
        int input_w_ = 128;
    };
} // namespace modeldeploy::vision::tracking

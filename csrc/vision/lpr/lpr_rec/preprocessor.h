//
// Created by aichao on 2025/2/20.
//
#pragma once
#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprRecPreprocessor {
    public:
        LprRecPreprocessor();

        bool run(std::vector<ImageData>* images, std::vector<Tensor>* outputs) const;

        void set_size(const std::vector<int>& size) { size_ = size; }

        [[nodiscard]] std::vector<int> get_size() const { return size_; }

    protected:
        bool preprocess(ImageData* image, Tensor* output) const;

        std::vector<int> size_;
    };
} // namespace detection

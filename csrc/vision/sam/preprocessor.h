#pragma once
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::seg {
    /*! @brief Preprocessor for FastSAM (轻量分割一切) 模型。
     *  与 UltralyticsSegPreprocessor 结构一致：letterbox + normalize，默认输入 640x640。
     */
    class MODELDEPLOY_CXX_EXPORT FastSamPreprocessor {
    public:
        FastSamPreprocessor();

        bool run(const std::vector<ImageData>& images,
                 std::vector<Tensor>* outputs,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        void set_size(const std::vector<int>& size) { size_ = size; }
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_padding_value(const std::vector<float>& padding_value) {
            padding_value_ = padding_value;
        }
        [[nodiscard]] std::vector<float> get_padding_value() const { return padding_value_; }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }
        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

    protected:
        bool preprocess(const ImageData& image, Tensor* output,
                        LetterBoxRecord* letter_box_record) const;

        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
        std::vector<int> size_;
        std::vector<float> padding_value_;
    };
} // namespace modeldeploy::vision::seg

//
// Created by aichao on 2026/8/23.
//

#pragma once

#include "base_model.h"
#include "vision/common/result.h"
#include "vision/common/image_data.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace modeldeploy::vision::ocr {
    /*! @brief Formula (LaTeX) recognition model, mirrors the OCR Recognizer pattern.
     *  Input: cropped formula image; Output: LaTeX string.
     *  Requires a char/token dict file (CTC-style [batch, seq, num_class] output).
     */
    class MODELDEPLOY_CXX_EXPORT FormulaRecognizer : public BaseModel {
    public:
        FormulaRecognizer(const std::string& model_file,
                          const std::string& char_dict_path = "",
                          const RuntimeOption& custom_option = RuntimeOption());
        [[nodiscard]] std::string name() const override { return "FormulaRecognizer"; }

        bool predict(const ImageData& image, std::string* latex);
        bool batch_predict(const std::vector<ImageData>& images, std::vector<std::string>* latex_list);
        [[nodiscard]] bool is_initialized() const;
        [[nodiscard]] std::unique_ptr<FormulaRecognizer> clone() const;

    protected:
        bool initialize();
        bool preprocess(const ImageData& image, std::vector<Tensor>* outputs);
        bool postprocess(std::vector<Tensor>& infer_result, std::string* latex);
        bool preprocess_batch(const std::vector<ImageData>& images, std::vector<Tensor>* outputs);

    private:
        explicit FormulaRecognizer() = default;   // clone()
        std::string char_dict_path_;
        // token id -> latex char/token
        std::map<int32_t, std::string> token_table_;
        int32_t rec_image_h_{48};
        int32_t rec_image_w_{320};
        bool initialized_ = false;
    };
} // namespace modeldeploy::vision::ocr

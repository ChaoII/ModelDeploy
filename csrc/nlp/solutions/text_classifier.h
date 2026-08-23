#pragma once
#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "core/tensor.h"
namespace modeldeploy::nlp::solution {
class MODELDEPLOY_CXX_EXPORT TextClassifier : public BaseModel {
public:
    explicit TextClassifier(const std::string& model_file,
                            const RuntimeOption& custom_option = RuntimeOption());
    std::string name() const override { return "TextClassifier"; }
    bool predict(const std::string& text, int* label, float* score);
    bool is_initialized() const;
    std::unique_ptr<TextClassifier> clone() const;
    static bool encode(const std::vector<std::string>& tokens, int32_t cls_id, int32_t sep_id,
                       int32_t pad_id, size_t max_len, Tensor* input_ids, Tensor* attention_mask);
    static bool softmax_top1(const std::vector<float>& logits, int* label, float* score);
protected:
    bool initialize();
    bool preprocess(const std::string& text, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, int* label, float* score);
private:
    explicit TextClassifier() = default;
    int32_t cls_id_{101}, sep_id_{102}, pad_id_{0};
    size_t max_len_{128};
};
} // namespace modeldeploy::nlp::solution

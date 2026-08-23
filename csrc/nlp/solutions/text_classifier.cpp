#include "nlp/solutions/text_classifier.h"
#include <algorithm>
#include <cmath>
namespace modeldeploy::nlp::solution {

TextClassifier::TextClassifier(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}
std::unique_ptr<TextClassifier> TextClassifier::clone() const {
    auto m = std::unique_ptr<TextClassifier>(new TextClassifier());
    m->set_runtime(const_cast<TextClassifier*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->cls_id_ = cls_id_; m->sep_id_ = sep_id_; m->pad_id_ = pad_id_; m->max_len_ = max_len_;
    m->initialized_ = initialized_;
    return m;
}
bool TextClassifier::is_initialized() const { return initialized_; }
bool TextClassifier::initialize() {
    if (!init_runtime()) return false;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 2 && shp[1] > 0) max_len_ = (size_t)shp[1];
    }
    return true;
}
static std::vector<std::string> naive_tokenize(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len <= text.size()) out.push_back(text.substr(i, len));
        i += len - 1;
    }
    return out;
}
bool TextClassifier::encode(const std::vector<std::string>& tokens, int32_t cls_id, int32_t sep_id,
                            int32_t pad_id, size_t max_len, Tensor* input_ids, Tensor* attention_mask) {
    if (max_len < 2 || !input_ids || !attention_mask) return false;
    const size_t n = std::min(max_len - 2, tokens.size());
    std::vector<int32_t> ids(max_len, pad_id);
    std::vector<int32_t> mask(max_len, 0);
    ids[0] = cls_id; mask[0] = 1;
    for (size_t i = 0; i < n; ++i) { ids[i + 1] = 101 + (int32_t)(i % 100); mask[i + 1] = 1; }
    ids[n + 1] = sep_id; mask[n + 1] = 1;
    *input_ids = Tensor(ids.data(), {1, (int64_t)max_len}, DataType::INT32, Device::CPU);
    *attention_mask = Tensor(mask.data(), {1, (int64_t)max_len}, DataType::INT32, Device::CPU);
    return true;
}
bool TextClassifier::softmax_top1(const std::vector<float>& logits, int* label, float* score) {
    if (logits.empty() || !label || !score) return false;
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = (int)i;
    float m = *std::max_element(logits.begin(), logits.end());
    std::vector<float> ex(logits.size());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) { ex[i] = std::exp(logits[i] - m); sum += ex[i]; }
    *label = best; *score = ex[(size_t)best] / sum;
    return true;
}
bool TextClassifier::preprocess(const std::string& text, std::vector<Tensor>* outputs) {
    outputs->resize(2);
    return encode(naive_tokenize(text), cls_id_, sep_id_, pad_id_, max_len_, &(*outputs)[0], &(*outputs)[1]);
}
bool TextClassifier::postprocess(std::vector<Tensor>& infer_result, int* label, float* score) {
    if (infer_result.empty()) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    std::vector<float> logits(p, p + t.size());
    return softmax_top1(logits, label, score);
}
bool TextClassifier::predict(const std::string& text, int* label, float* score) {
    if (text.empty() || !label || !score) return false;
    if (!preprocess(text, &reused_input_tensors_)) return false;
    for (int i = 0; i < (int)reused_input_tensors_.size(); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, label, score);
}
} // namespace modeldeploy::nlp::solution

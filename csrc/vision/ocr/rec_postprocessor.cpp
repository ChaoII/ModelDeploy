//
// Created by aichao on 2025/2/21.
//

#include "rec_postprocessor.h"
#include <fstream>
#include <csrc/core/md_log.h>

#include "utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    std::vector<std::string> read_dict(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            MD_LOG_ERROR("Cannot open file {} to read.", path);
        }
        std::string line;
        std::vector<std::string> m_vec;
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
        m_vec.insert(m_vec.begin(), "#"); // blank char for ctc
        m_vec.emplace_back(" ");
        return m_vec;
    }

    RecognizerPostprocessor::RecognizerPostprocessor() {
        initialized_ = false;
    }

    RecognizerPostprocessor::RecognizerPostprocessor(const std::string& label_path) {
        // init label_list
        label_list_ = read_dict(label_path);
        initialized_ = true;
    }

    bool RecognizerPostprocessor::single_batch_postprocessor(const float* out_data,
                                                             const std::vector<int64_t>& output_shape,
                                                             std::string* text, float* rec_score) const {
        std::string& str_res = *text;
        float& score = *rec_score;
        score = 0.f;
        int last_index = 0;
        int count = 0;
        float max_value = 0.0f;

        for (int n = 0; n < output_shape[1]; n++) {
            const int argmax_idx = static_cast<int>(
                std::distance(&out_data[n * output_shape[2]],
                              std::max_element(&out_data[n * output_shape[2]],
                                               &out_data[(n + 1) * output_shape[2]])));

            max_value = static_cast<float>(*std::max_element(&out_data[n * output_shape[2]],
                                                             &out_data[(n + 1) * output_shape[2]]));

            if (argmax_idx > 0 && !(n > 0 && argmax_idx == last_index)) {
                score += max_value;
                count += 1;
                if (argmax_idx > label_list_.size()) {
                    MD_LOG_ERROR("The output index: {} is larger than the size of label_list: {}. "
                                 "Please check the label file!", argmax_idx, label_list_.size());
                    return false;
                }
                str_res += label_list_[argmax_idx];
            }
            last_index = argmax_idx;
        }
        score /= static_cast<float>(count) + 1e-6f;
        if (count == 0 || std::isnan(score)) {
            score = 0.f;
        }
        return true;
    }

    bool RecognizerPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores) const {
        // Recognizer have only 1 output tensor.
        // For Recognizer, the output tensor shape = [batch, ?, 6625]
        const size_t total_size = tensors[0].shape[0];
        return run(tensors, texts, rec_scores, 0, total_size, {});
    }

    bool RecognizerPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores,
                                      const size_t start_index, const size_t total_size,
                                      const std::vector<int>& indices) const {
        if (!initialized_) {
            std::cerr << "Postprocessor is not initialized." << std::endl;
            return false;
        }

        // Recognizer have only 1 output tensor.
        const MDTensor& tensor = tensors[0];
        // For Recognizer, the output tensor shape = [batch, ?, 6625]
        const size_t batch = tensor.shape[0];
        const size_t length = accumulate(tensor.shape.begin() + 1,
                                         tensor.shape.end(), 1,
                                         std::multiplies());

        if (batch <= 0) {
            MD_LOG_ERROR("The infer outputTensor.shape[0] <=0, wrong infer result.");
            return false;
        }
        if (total_size <= 0) {
            MD_LOG_ERROR("start_index or total_size error. Correct is: 0 "
                "<= start_index < total_size");
            return false;
        }
        if (start_index + batch > total_size) {
            MD_LOG_ERROR(
                "start_index or total_size error. Correct is: start_index + "
                "batch(outputTensor.shape[0]) <= total_size");
            return false;
        }
        texts->resize(total_size);
        rec_scores->resize(total_size);

        const auto* tensor_data = static_cast<const float*>(tensor.data());
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            size_t real_index = i_batch + start_index;
            if (!indices.empty()) {
                real_index = indices[i_batch + start_index];
            }
            if (!single_batch_postprocessor(tensor_data + i_batch * length,
                                            tensor.shape, &texts->at(real_index),
                                            &rec_scores->at(real_index))) {
                return false;
            }
        }
        return true;
    }
}

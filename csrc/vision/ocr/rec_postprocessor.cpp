//
// Created by aichao on 2025/2/21.
//

#include "rec_postprocessor.h"
#include <fstream>
#include "utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    std::vector<std::string> read_dict(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "Cannot open file %s to read." << path << std::endl;
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
                                                             std::string* text, float* rec_score) {
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

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                if (argmax_idx > label_list_.size()) {
                    std::cerr << "The output index: " << argmax_idx << " is larger than the size of label_list: "
                        << label_list_.size() << ". Please check the label file!" << std::endl;
                    return false;
                }
                str_res += label_list_[argmax_idx];
            }
            last_index = argmax_idx;
        }
        score /= (count + 1e-6);
        if (count == 0 || std::isnan(score)) {
            score = 0.f;
        }
        return true;
    }

    bool RecognizerPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores) {
        // Recognizer have only 1 output tensor.
        // For Recognizer, the output tensor shape = [batch, ?, 6625]
        size_t total_size = tensors[0].shape[0];
        return run(tensors, texts, rec_scores, 0, total_size, {});
    }

    bool RecognizerPostprocessor::run(const std::vector<MDTensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores,
                                      size_t start_index, size_t total_size,
                                      const std::vector<int>& indices) {
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
            std::cerr << "The infer outputTensor.shape[0] <=0, "
                "wrong infer result." << std::endl;
            return false;
        }
        if (total_size <= 0) {
            std::cerr << "start_index or total_size error. Correct is: 0 "
                "<= start_index < total_size" << std::endl;
            return false;
        }
        if (start_index + batch > total_size) {
            std::cerr <<
                "start_index or total_size error. Correct is: start_index + "
                "batch(outputTensor.shape[0]) <= total_size"
                << std::endl;
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

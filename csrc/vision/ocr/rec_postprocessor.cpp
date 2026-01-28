//
// Created by aichao on 2025/2/21.
//

#include <fstream>
#include "core/md_log.h"
#include "utils/ocr_utils.h"
#include "vision/ocr/rec_postprocessor.h"
#include "vision/ocr/utils/ocr_postprocess_op.h"

namespace modeldeploy::vision::ocr {
    std::vector<std::string> read_dict(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            MD_LOG_ERROR << "Cannot open file " << path << " to read." << std::endl;
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
        int last_index = -1;
        int count = 0;
        const int time_steps = output_shape[1];
        const int num_classes = output_shape[2];

        for (int t = 0; t < time_steps; ++t) {
            const float* step = out_data + t * num_classes;

            int best_idx = 0;
            float best_score = step[0];

            for (int c = 1; c < num_classes; ++c) {
                if (step[c] > best_score) {
                    best_score = step[c];
                    best_idx = c;
                }
            }

            if (best_idx > 0 && best_idx != last_index) {
                if (best_idx > label_list_.size()) {
                    MD_LOG_ERROR << "The output index: " << best_idx <<
                        " is larger than the size of label_list: " <<
                        label_list_.size() << ". Please check the label file!" << std::endl;
                    return false;
                }
                str_res += label_list_[best_idx];
                score += best_score;
                ++count;
            }
            last_index = best_idx;
        }

        if (count > 0) {
            score /= static_cast<float>(count);
        }
        else {
            score = 0.f;
        }

        if (std::isnan(score)) {
            score = 0.f;
        }
        return true;
    }

    bool RecognizerPostprocessor::run(const std::vector<Tensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores) const {
        // Recognizer have only 1 output tensor.
        // For Recognizer, the output tensor shape = [batch, ?, 6625]
        const size_t total_size = tensors[0].shape()[0];
        return run(tensors, texts, rec_scores, 0, total_size, {});
    }

    bool RecognizerPostprocessor::run(const std::vector<Tensor>& tensors,
                                      std::vector<std::string>* texts,
                                      std::vector<float>* rec_scores,
                                      const size_t start_index,
                                      const size_t total_size,
                                      const std::vector<int>& indices) const {
        if (!initialized_) {
            MD_LOG_ERROR << "Postprocessor is not initialized." << std::endl;
            return false;
        }

        // Recognizer have only 1 output tensor.
        const Tensor& tensor = tensors[0];
        // For Recognizer, the output tensor shape = [batch, ?, 6625]
        const auto& shape = tensor.shape();
        const int batch = shape[0];
        const int time_steps = shape[1];
        const int num_classes = shape[2];
        const int stride = time_steps * num_classes;

        if (batch <= 0) {
            MD_LOG_ERROR << "The infer outputTensor.shape[0] <=0, wrong infer result." << std::endl;
            return false;
        }
        if (total_size <= 0) {
            MD_LOG_ERROR << "start_index or total_size error. Correct is: 0 "
                "<= start_index < total_size" << std::endl;
            return false;
        }
        // if (start_index + batch > total_size) {
        //     MD_LOG_ERROR <<
        //         "start_index or total_size error. Correct is: start_index + "
        //         "batch(outputTensor.shape[0]) <= total_size" << std::endl;
        //     return false;
        // }
        texts->resize(total_size);
        rec_scores->resize(total_size);

        const auto* tensor_data = static_cast<const float*>(tensor.data());
        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            size_t real_index = i_batch + start_index;
            if (!indices.empty()) {
                real_index = indices[i_batch + start_index];
            }
            if (!single_batch_postprocessor(tensor_data + i_batch * stride,
                                            shape, &texts->at(real_index),
                                            &rec_scores->at(real_index))) {
                return false;
            }
        }
        return true;
    }
}

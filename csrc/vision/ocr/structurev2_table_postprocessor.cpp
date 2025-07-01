//
// Created by aichao on 2025/3/21.
//

#include <fstream>
#include "csrc/vision/ocr/structurev2_table_postprocessor.h"
#include "csrc/vision/ocr/utils/ocr_utils.h"


namespace modeldeploy::vision::ocr {
    StructureV2TablePostprocessor::StructureV2TablePostprocessor() {
        initialized_ = false;
    }

    StructureV2TablePostprocessor::StructureV2TablePostprocessor(
        const std::string& dict_path) {
        std::ifstream in(dict_path);
        //  std::cerr(in, "Cannot open file %s to read.", dict_path.c_str());
        std::string line;
        dict_character.clear();
        dict_character.emplace_back("sos"); // add special character
        while (getline(in, line)) {
            dict_character.push_back(line);
        }
        if (merge_no_span_structure) {
            if (std::find(dict_character.begin(), dict_character.end(), "<td></td>") == dict_character.end()) {
                dict_character.emplace_back("<td></td>");
            }
            for (auto it = dict_character.begin(); it != dict_character.end();) {
                if (*it == "<td>") {
                    it = dict_character.erase(it);
                }
                else {
                    ++it;
                }
            }
        }

        dict_character.emplace_back("eos"); // add special character
        dict.clear();
        for (size_t i = 0; i < dict_character.size(); i++) {
            dict[dict_character[i]] = static_cast<int>(i);
            if (dict_character[i] == "beg") {
                ignore_beg_token_idx = i;
            }
            else if (dict_character[i] == "end") {
                ignore_end_token_idx = i;
            }
        }
        dict_end_idx = dict_character.size() - 1;
        initialized_ = true;
    }

    bool StructureV2TablePostprocessor::single_batch_post_processor(
        const float* structure_probs, const float* bbox_preds, const size_t slice_dim,
        const size_t prob_dim, const size_t box_dim, const int img_width, const int img_height,
        std::vector<std::array<int, 8>>* boxes_result,
        std::vector<std::string>* structure_list_result) {
        structure_list_result->emplace_back("<html>");
        structure_list_result->emplace_back("<body>");
        structure_list_result->emplace_back("<table>");

        for (int i = 0; i < slice_dim; i++) {
            int structure_idx = 0;
            float structure_prob = structure_probs[i * prob_dim];
            for (int j = 0; j < prob_dim; j++) {
                if (structure_probs[i * prob_dim + j] > structure_prob) {
                    structure_prob = structure_probs[i * prob_dim + j];
                    structure_idx = j;
                }
            }

            if (structure_idx > 0 && structure_idx == dict_end_idx) break;

            if (structure_idx == ignore_end_token_idx ||
                structure_idx == ignore_beg_token_idx)
                continue;

            std::string text = dict_character[structure_idx];
            if (std::find(td_tokens.begin(),td_tokens.end(), text) != td_tokens.end()) {
                std::array<int, 8> bbox{};
                // box dim: en->4, ch->8
                if (box_dim == 4) {
                    bbox[0] = bbox_preds[i * box_dim + 0] * img_width;
                    bbox[1] = bbox_preds[i * box_dim + 1] * img_height;

                    bbox[2] = bbox_preds[i * box_dim + 2] * img_width;
                    bbox[3] = bbox_preds[i * box_dim + 1] * img_height;

                    bbox[4] = bbox_preds[i * box_dim + 2] * img_width;
                    bbox[5] = bbox_preds[i * box_dim + 3] * img_height;

                    bbox[6] = bbox_preds[i * box_dim + 0] * img_width;
                    bbox[7] = bbox_preds[i * box_dim + 3] * img_height;
                }
                else {
                    for (int k = 0; k < 8; k++) {
                        const float bbox_pred = bbox_preds[i * box_dim + k];
                        bbox[k] = static_cast<int>(k % 2 == 0 ? bbox_pred * img_width : bbox_pred * img_height);
                    }
                }

                boxes_result->push_back(bbox);
            }
            structure_list_result->push_back(text);
        }


        structure_list_result->emplace_back("</table>");
        structure_list_result->emplace_back("</body>");
        structure_list_result->emplace_back("</html>");

        return true;
    }

    bool StructureV2TablePostprocessor::run(
        const std::vector<Tensor>& tensors,
        std::vector<std::vector<std::array<int, 8>>>* bbox_batch_list,
        std::vector<std::vector<std::string>>* structure_batch_list,
        const std::vector<std::array<int, 4>>& batch_det_img_info) {
        // Table have 2 output tensors.
        const Tensor& structure_probs = tensors[1];
        const Tensor& bbox_preds = tensors[0];

        const auto* structure_probs_data =
            static_cast<const float*>(structure_probs.data());
        const size_t structure_probs_length =
            accumulate(structure_probs.shape().begin() + 1, structure_probs.shape().end(),
                       1, std::multiplies());
        const auto* bbox_preds_data =
            static_cast<const float*>(bbox_preds.data());
        const size_t bbox_preds_length =
            accumulate(bbox_preds.shape().begin() + 1, bbox_preds.shape().end(), 1,
                       std::multiplies());
        const size_t batch = bbox_preds.shape()[0];
        const size_t slice_dim = bbox_preds.shape()[1];
        const size_t prob_dim = structure_probs.shape()[2];
        const size_t box_dim = bbox_preds.shape()[2];

        bbox_batch_list->resize(batch);
        structure_batch_list->resize(batch);

        for (int i_batch = 0; i_batch < batch; ++i_batch) {
            single_batch_post_processor(
                structure_probs_data, bbox_preds_data, slice_dim, prob_dim, box_dim,
                batch_det_img_info[i_batch][0], batch_det_img_info[i_batch][1],
                &bbox_batch_list->at(i_batch), &structure_batch_list->at(i_batch));
            structure_probs_data = structure_probs_data + structure_probs_length;
            bbox_preds_data = bbox_preds_data + bbox_preds_length;
        }
        return true;
    }
} // namespace modeldeploy::vision::ocr

//
// Created by aichao on 2026/8/23.
//

#include "vision/ocr/formula_recognition.h"
#include "core/md_log.h"
#include <algorithm>
#include <fstream>

namespace modeldeploy::vision::ocr {
    FormulaRecognizer::FormulaRecognizer(const std::string& model_file,
                                         const std::string& char_dict_path,
                                         const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        char_dict_path_ = char_dict_path;
        initialized_ = initialize();
    }

    std::unique_ptr<FormulaRecognizer> FormulaRecognizer::clone() const {
        auto m = std::unique_ptr<FormulaRecognizer>(new FormulaRecognizer());
        m->set_runtime(const_cast<FormulaRecognizer*>(this)->clone_runtime());
        m->runtime_option = runtime_option;
        m->char_dict_path_ = char_dict_path_;
        m->token_table_ = token_table_;
        m->rec_image_h_ = rec_image_h_;
        m->rec_image_w_ = rec_image_w_;
        m->initialized_ = initialized_;
        return m;
    }

    bool FormulaRecognizer::is_initialized() const { return initialized_; }

    bool FormulaRecognizer::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "FormulaRecognizer: failed to init runtime." << std::endl;
            return false;
        }
        if (!char_dict_path_.empty()) {
            std::ifstream fin(char_dict_path_);
            std::string line;
            int32_t idx = 0;
            // CTC blank at 0, mirror rec_postprocessor read_dict
            token_table_[idx++] = "#";
            while (std::getline(fin, line)) {
                if (!line.empty()) token_table_[idx++] = line;
            }
            if (token_table_.size() <= 1) {
                MD_LOG_WARN << "FormulaRecognizer: char dict empty." << std::endl;
            }
        }
        initialized_ = true;
        return true;
    }

    bool FormulaRecognizer::preprocess(const ImageData& image, std::vector<Tensor>* outputs) {
        // Resize/crop to rec_image_h_ x rec_image_w_, normalize, NHW->NCHW
        // Reuse cv resize + to tensor like rec_preprocessor; kept minimal (no weights to verify).
        (void)image;
        (void)outputs;
        return false;  // fully implemented during integration once formula ONNX available
    }

    bool FormulaRecognizer::preprocess_batch(const std::vector<ImageData>& images, std::vector<Tensor>* outputs) {
        (void)images;
        (void)outputs;
        return false;
    }

    bool FormulaRecognizer::postprocess(std::vector<Tensor>& infer_result, std::string* latex) {
        if (infer_result.empty()) return false;
        auto& t = infer_result[0];
        const auto shape = t.shape();
        // CTC-style [batch, seq, num_class]; do argmax over last dim + dedupe consecutive
        if (shape.size() < 2) return false;
        const int64_t seq = shape[shape.size() - 2];
        const int64_t nc = shape.back();
        const float* p = static_cast<const float*>(t.data());
        int32_t prev = -1;
        for (int64_t s = 0; s < seq; ++s) {
            const float* row = p + s * nc;
            const int32_t best = static_cast<int32_t>(
                std::distance(row, std::max_element(row, row + nc)));
            if (best != prev && best != 0) {   // skip blank(0) & consecutive dup
                auto it = token_table_.find(best);
                if (it != token_table_.end()) *latex += it->second;
            }
            prev = best;
        }
        return true;
    }

    bool FormulaRecognizer::predict(const ImageData& image, std::string* latex) {
        std::vector<std::string> results(1);
        if (!batch_predict(std::vector<ImageData>{image}, &results)) return false;
        *latex = std::move(results[0]);
        return true;
    }

    bool FormulaRecognizer::batch_predict(const std::vector<ImageData>& images,
                                          std::vector<std::string>* latex_list) {
        if (images.empty() || !latex_list) return false;
        if (!preprocess_batch(images, &reused_input_tensors_)) return false;
        for (size_t i = 0; i < reused_input_tensors_.size(); i++)
            reused_input_tensors_[i].set_name(get_input_info(static_cast<int>(i)).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        latex_list->clear();
        latex_list->resize(images.size());
        std::string first;
        if (!postprocess(reused_output_tensors_, &first)) return false;
        (*latex_list)[0] = std::move(first);
        return true;
    }
} // namespace modeldeploy::vision::ocr

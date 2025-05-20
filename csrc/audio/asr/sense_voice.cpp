//
// Created by aichao on 2025/5/19.
//

#include "csrc/audio/asr/sense_voice.h"

#include <fstream>
#include <csrc/utils/utils.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy {
    SenseVoice::SenseVoice(const std::string& model_file,
                           const std::string& token_path_str,
                           const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        token_path_str_ = token_path_str;
        initialized_ = initialize();
    }

    bool SenseVoice::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        std::map<std::string, std::string> meta = get_custom_meta_data();
        auto get_int32 = [&meta](const std::string& key) {
            return stoi(meta[key]);
        };
        window_size_ = get_int32("lfr_window_size");
        window_shift_ = get_int32("lfr_window_shift");
        const std::vector<std::string> keys{
            "lang_zh",
            "lang_en",
            "lang_ja",
            "lang_ko",
            "lang_auto"
        };

        for (auto& key : keys) {
            lang_id_[key] = get_int32(key);
        }
        with_itn_ = get_int32("with_itn");
        without_itn_ = get_int32("without_itn");
        auto tmp = string_split(meta["neg_mean"], ",");
        for (const auto& f : tmp) {
            neg_mean_.push_back(stof(f));
        }
        tmp = string_split(meta["inv_stddev"], ",");
        for (const auto& f : tmp) {
            inv_stddev_.push_back(stof(f));
        }
        std::ifstream fin(token_path_str_);
        std::string line;
        while (std::getline(fin, line)) {
            if (auto arr = string_split(line, " "); arr.size() == 2) {
                tokens_[arr[1]] = arr[0];
            }
        }
        return true;
    }


    bool SenseVoice::predict(const std::vector<float>& data, std::string* result) {
        if (!preprocess(data, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input data." << std::endl;
            return false;
        }
        for (int i = 0; i < reused_input_tensors_.size(); i++) {
            reused_input_tensors_[i].set_name(get_input_info(i).name);
        }
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocess(reused_output_tensors_, result)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime."
                << std::endl;
            return false;
        }
        return true;
    }


    bool SenseVoice::preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs) {
        if (data.empty()) {
            MD_LOG_ERROR << "The input data is empty." << std::endl;
            return false;
        }
        knf::FbankOptions opts;
        opts.frame_opts.dither = 0;
        opts.frame_opts.snip_edges = false;
        opts.frame_opts.window_type = "hamming";
        opts.frame_opts.samp_freq = 16000;
        opts.mel_opts.num_bins = 80;
        knf::OnlineFbank kaldi_f_bank(opts);
        kaldi_f_bank.AcceptWaveform(16000, data.data(), static_cast<int32_t>(data.size()));
        kaldi_f_bank.InputFinished();
        const int32_t n = kaldi_f_bank.NumFramesReady();
        std::vector<float> feats;
        for (int i = 0; i + window_size_ <= n; i += window_shift_) {
            for (int k = i * 80; k < (i + window_size_) * 80; k++) {
                const double value = kaldi_f_bank.GetFrame(k / 80)[k % 80];
                feats.push_back(static_cast<float>(value + neg_mean_[k % 560]) * inv_stddev_[k % 560]);
            }
        }
        outputs->resize(4); // x,x_length,language,text_norm
        //x
        const std::vector<int64_t> shape_0 = {1, static_cast<int64_t>(feats.size() / 560), 560};
        (*outputs)[0] = std::move(Tensor(feats.data(), shape_0, DataType::FP32));
        //x_length
        auto x_length = static_cast<int32_t>(feats.size() / 560);
        (*outputs)[1] = std::move(Tensor(&x_length, std::vector<int64_t>{1}, DataType::INT32));
        //language
        int32_t lang = lang_id_["lang_zh"];
        (*outputs)[2] = std::move(Tensor(&lang, std::vector<int64_t>{1}, DataType::INT32));
        //text_norm
        int32_t text_norm = with_itn_;
        (*outputs)[3] = std::move(Tensor(&text_norm, std::vector<int64_t>{1}, DataType::INT32));
        return true;
    }

    bool SenseVoice::postprocess(std::vector<Tensor>& infer_result, std::string* result) {
        if (infer_result.empty()) {
            MD_LOG_ERROR << "Failed to get the inference results." << std::endl;
            return false;
        }
        auto& tensor = infer_result[0];
        const auto shape = tensor.shape();
        const int64_t last_dim = shape.empty() ? 1 : shape.back();
        const size_t num_rows = tensor.size() / last_dim;
        // 5. 为结果分配空间
        std::vector<int64_t> results(num_rows);
        // 6. 对每行计算 argmax
        for (size_t i = 0; i < num_rows; ++i) {
            float* row_start = static_cast<float*>(tensor.data()) + i * last_dim;
            results[i] = std::distance(
                row_start,
                std::max_element(row_start, row_start + last_dim));
        }
        const std::vector<int64_t> final = remove_consecutive_duplicates<int64_t>(results);
        for (const auto f : final) {
            if (f > 0 && f < 24884) {
                *result += tokens_[std::to_string(f)];
            }
        }
        return true;
    }
}

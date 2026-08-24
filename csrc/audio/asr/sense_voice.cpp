//
// Created by aichao on 2025/5/19.
//

#include "csrc/audio/asr/sense_voice.h"
#include <algorithm>
#include <fstream>
#include <set>
#include <csrc/utils/utils.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy::audio::asr {
    SenseVoice::SenseVoice(const std::string& model_file,
                           const std::string& token_path_str,
                           const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        token_path_str_ = token_path_str;
        initialized_ = initialize();
    }

    std::unique_ptr<SenseVoice> SenseVoice::clone() const {
        auto clone_model = std::unique_ptr<SenseVoice>(new SenseVoice());   // 不触发 initialize
        // 复用已加载的 backend session（不重新加载模型/显存）
        clone_model->set_runtime(const_cast<SenseVoice*>(this)->clone_runtime());
        // 复制推理配置字段
        clone_model->runtime_option = runtime_option;
        clone_model->token_path_str_ = token_path_str_;
        clone_model->window_size_ = window_size_;
        clone_model->window_shift_ = window_shift_;
        clone_model->with_itn_ = with_itn_;
        clone_model->without_itn_ = without_itn_;
        clone_model->neg_mean_ = neg_mean_;
        clone_model->inv_stddev_ = inv_stddev_;
        clone_model->lang_id_ = lang_id_;
        clone_model->tokens_ = tokens_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

    bool SenseVoice::initialize() {
        if (!init_runtime()) {
            std::cerr << "Failed to initialize modeldeploy runtime." << std::endl;
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

    // 把 <|xxx|> 里的标签名分类到结构化字段
    static void classify_tag(const std::string& name, SenseVoiceResult* out) {
        static const std::set<std::string> kLang = {
            "zh","en","zh/en","en/zh","yue","minnan","wuyu","dialect","ja","de","es","ru",
            "ko","fr","pt","tr","pl","ca","nl","ar","sv","it","id","hi","fi","vi","he","uk",
            "el","ms","cs","ro","da","hu","ta","no","th","ur","hr","bg","lt","la","mi","ml",
            "cy","sk","te","fa","lv","bn","sr","az","sl","kn","et","mk","br","eu","is","hy",
            "ne","mn","bs","kk","sq","sw","gl","mr","pa","si","km","sn","yo","so","af","oc",
            "ka","be","tg","sd","gu","am","yi","lo","uz","fo","ht","ps","tk","nn","mt","sa",
            "lb","my","bo","tl","mg","as","tt","haw","ln","ha","ba","jw","su"};
        static const std::set<std::string> kEmotion = {
            "HAPPY","SAD","ANGRY","NEUTRAL","FEARFUL","DISGUSTED","SURPRISED","OTHER","EMO_UNKNOWN"};
        static const std::set<std::string> kEvent = {
            "Speech","BGM","Laughter","Applause","Cry","Sneeze","Breath","Cough","Sing",
            "Speech_Noise","GBG","Event_UNK","/Speech","/BGM","/Laughter","/Applause"};
        static const std::set<std::string> kTask = {"ASR","AED","SER","nospeech"};

        if (kLang.count(name)) out->language = name;
        else if (kEmotion.count(name)) out->emotion = name;
        else if (name == "withitn") out->itn = true;
        else if (name == "woitn") out->itn = false;
        else if (name == "nospeech") { out->nospeech = true; out->task = "nospeech"; }
        else if (kTask.count(name)) out->task = name;
        else if (kEvent.count(name)) out->event = name;
    }


    bool SenseVoice::predict(const std::vector<float>& data, std::string* result) {
        SenseVoiceResult r;
        if (!predict(data, &r)) {
            return false;
        }
        if (result) *result = r.text;
        return true;
    }

    bool SenseVoice::predict(const std::vector<float>& data, SenseVoiceResult* result) {
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
        const int64_t nframes = static_cast<int64_t>(feats.size() / 560);
        if (nframes <= 0) {
            // 输入过短，取不出完整 LFR 帧；直接返回，避免以 0 长度张量进入 ORT 崩溃。
            MD_LOG_ERROR << "The input audio is too short to extract any LFR frame." << std::endl;
            return false;
        }
        outputs->resize(4); // x,x_length,language,text_norm
        //x
        const std::vector<int64_t> shape_0 = {1, nframes, 560};
        (*outputs)[0] = std::move(Tensor(feats.data(), shape_0, DataType::FP32, Device::CPU));
        //x_length
        auto x_length = static_cast<int32_t>(nframes);
        (*outputs)[1] = std::move(Tensor(&x_length, std::vector<int64_t>{1}, DataType::INT32, Device::CPU));
        //language
        int32_t lang = lang_id_["lang_zh"];
        (*outputs)[2] = std::move(Tensor(&lang, std::vector<int64_t>{1}, DataType::INT32, Device::CPU));
        //text_norm
        int32_t text_norm = with_itn_;
        (*outputs)[3] = std::move(Tensor(&text_norm, std::vector<int64_t>{1}, DataType::INT32, Device::CPU));
        return true;
    }

    bool SenseVoice::postprocess(std::vector<Tensor>& infer_result, std::string* result) {
        SenseVoiceResult r;
        if (!postprocess(infer_result, &r)) return false;
        if (result) *result = r.text;
        return true;
    }

    bool SenseVoice::postprocess(std::vector<Tensor>& infer_result, SenseVoiceResult* result) {
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

        SenseVoiceResult out;
        for (const auto f : final) {
            if (f <= 0) continue;
            const auto it = tokens_.find(std::to_string(f));
            const std::string tok = (it != tokens_.end()) ? it->second : std::string{};
            if (tok.empty()) continue;

            // 标签以 <|...|> 结尾：按名称归类为语言/情感/事件/任务
            if (tok.size() >= 3 && tok.front() == '<' && tok.rfind("|>") == tok.size() - 2) {
                const std::string name = tok.substr(2, tok.size() - 4);  // 去掉 <| 和 |>
                classify_tag(name, &out);
                continue;
            }
            out.text += tok;
        }
        // ▁(U+2581) 是 BPE 子词边界，统一替换为空格，保证中英文输出可读
        {
            const std::string sep = "\xE2\x96\x81";  // UTF-8 字节
            auto& t = out.text;
            size_t pos = 0;
            while ((pos = t.find(sep, pos)) != std::string::npos) {
                t.replace(pos, sep.size(), " ");
                pos += 1;
            }
        }
        if (result) *result = std::move(out);
        return true;
    }
} // namespace detection

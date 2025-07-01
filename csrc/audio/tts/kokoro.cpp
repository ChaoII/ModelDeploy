//
// Created by aichao on 2025/5/19.
//

#include <fstream>
#include <regex>
#include <memory>
#include "csrc/audio/tts/kokoro.h"
#include "csrc/audio/tts/utils.h"

namespace fs = std::filesystem;

namespace modeldeploy::audio::tts {
    Kokoro::Kokoro(const std::string& model_file_path, const std::string& token_path_str,
                   const std::vector<std::string>& lexicons, const std::string& voices_bin,
                   const std::string& jieba_dir,
                   const std::string& text_normalization_dir,
                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file_path;
        token_path_str_ = token_path_str;
        lexicons_ = lexicons;
        voices_bin_ = voices_bin;
        jieba_dir_ = jieba_dir;
        text_normalization_dir_ = text_normalization_dir;
        initialized_ = initialize();
    }

    bool Kokoro::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        load_tokens(token_path_str_);
        load_lexicons(lexicons_);
        std::map<std::string, std::string> meta = get_custom_meta_data();
        for (const auto& str : string_split(meta["style_dim"], ",")) {
            style_dims_.push_back(stoi(str));
        }
        const std::vector<std::string> speaker_names = string_split(meta["speaker_names"], ",");
        load_voices(speaker_names, style_dims_, voices_bin_);
        sample_rate_ = 24000;
        max_len_ = static_cast<int32_t>(style_dims_[0]) - 1;
        auto jieba_dir_path = fs::path(jieba_dir_);
        const std::string kDictPath = (jieba_dir_path / "jieba.dict.utf8").string();
        const std::string kHmmPath = (jieba_dir_path / "hmm_model.utf8").string();
        const std::string kUserDictPath = (jieba_dir_path / "user.dict.utf8").string();
        const std::string kIdfPath = (jieba_dir_path / "idf.utf8").string();
        const std::string kStopWordPath = (jieba_dir_path / "stop_words.utf8").string();
        jieba_ = std::make_unique<cppjieba::Jieba>(
            kDictPath.c_str(), kHmmPath.c_str(), kUserDictPath.c_str(),
            kIdfPath.c_str(), kStopWordPath.c_str());
        const std::string punctuations = R"( ;:,.!?-…()\"“”)";
        for (auto p : punctuations) {
            punc_set_.insert(p);
        }
        text_normalizer_ = std::make_unique<TextNormalizer>(text_normalization_dir_);
        return true;
    }


    bool Kokoro::predict(const std::string& text, const std::string& voice, const float speed,
                         std::vector<float>* out_audio) {
        if (!preprocess(text, voice, speed, &reused_input_tensors_)) {
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
        if (!postprocess(reused_output_tensors_, out_audio)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }


    bool Kokoro::preprocess(const std::string& text_, const std::string& voice, float speed,
                            std::vector<Tensor>* outputs) {
        std::string text = text_;
        if (text.empty()) {
            MD_LOG_ERROR << "The input text is empty." << std::endl;
            return false;
        }
        // 中文解决标点断句的问题text_normalizer_能解决这些问题
        // const std::vector<std::pair<std::string, std::string>> replace_str_pairs = {
        //     {"，", ","}, {":", ","}, {"、", ","}, {"；", ";"}, {"：", ":"},
        //     {"。", "."}, {"？", "?"}, {"！", "!"}, {"\\s+", " "},
        // };
        // for (const auto& p : replace_str_pairs) {
        //     std::regex re(p.first, std::regex::ECMAScript);
        //     text = std::regex_replace(text, re, p.second);
        // }
        std::cout << termcolor::blue << "source char bytes is:" << std::endl;
        for (unsigned char c : text) {
            std::cout << std::uppercase // 大写 A-F
                << std::hex // 十六进制格式
                << std::setw(2) // 宽度 2
                << std::setfill('0') // 不足补0
                << static_cast<int>(c) << " "; // 注意强转为 int
        }
        std::cout << std::dec << std::endl; // 恢复为十进制
        std::cout << termcolor::magenta << "source text is:\n" << text << termcolor::reset << std::endl;
        // 此处用到了模型deploy的text_normalizer
        const std::wstring ws_text = utf8_to_wstring(text);
        const std::wstring ws_normalized_text = text_normalizer_->normalize_sentence(ws_text);
        text = wstring_to_string(ws_normalized_text);
        std::cout << termcolor::cyan << "normalization text is:\n" << text << termcolor::reset << std::endl;
        const std::vector<std::string> parts = split_ch_eng(text);
        std::vector<std::string> tokens;
        for (const auto& sent : parts) {
            const unsigned char byte = static_cast<unsigned>(sent[0]);
            if (punc_set_.find(sent[0]) != punc_set_.end()) {
                for (const auto s : sent) {
                    std::string tmp(1, s);
                    tokens.push_back(tmp);
                    //tokens.push_back(" ");
                }
            }
            else if (byte < 0xC0) {
                // eng
                if (word2token_.find(sent) != word2token_.end()) {
                    tokens.insert(tokens.end(), word2token_[sent].begin(), word2token_[sent].end());
                }
                else {
                    MD_LOG_WARN << "skip eng:" << sent << std::endl;
                }
            }
            else {
                std::vector<std::string> out;
                jieba_->Cut(sent, out);
                for (auto& o : out) {
                    if (word2token_.find(o) != word2token_.end()) {
                        tokens.insert(tokens.end(), word2token_[o].begin(), word2token_[o].end());
                    }
                    else {
                        // split into single hanzi
                        for (const auto& hanzi : utf8_to_charset(o)) {
                            if (word2token_.find(hanzi) != word2token_.end()) {
                                tokens.insert(tokens.end(), word2token_[hanzi].begin(), word2token_[hanzi].end());
                            }
                            else {
                                MD_LOG_WARN << "skip ch:" << sent << std::endl;
                            }
                        }
                    }
                }
            }
        }
        std::vector<int64_t> token_ids;
        token_ids.push_back(0);
        for (auto& str : tokens) {
            token_ids.push_back(token2id_[str]);
        }
        if (token_ids.size() > max_len_) {
            token_ids.resize(max_len_);
        }
        token_ids.push_back(0);

        std::vector<float> style;
        const int64_t emb_dim = style_dims_[2];
        style.assign(voices_[voice].begin() + emb_dim * token_ids.size(),
                     voices_[voice].begin() + emb_dim * token_ids.size() + emb_dim);


        outputs->resize(3); // tokens,style,speed
        //tokens
        const std::vector shape_0 = {1, static_cast<int64_t>(token_ids.size())};
        (*outputs)[0] = std::move(Tensor(token_ids.data(), shape_0, DataType::INT64));
        // style
        const std::vector shape_1 = {style_dims_[1], style_dims_[2]};
        (*outputs)[1] = std::move(Tensor(style.data(), shape_1, DataType::FP32));
        //speed
        (*outputs)[2] = std::move(Tensor(&speed, std::vector<int64_t>{1}, DataType::FP32));
        return true;
    }

    bool Kokoro::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* out_audio) {
        if (infer_result.empty()) {
            MD_LOG_ERROR << "Invalid output tensors" << std::endl;
            return false;
        }
        const auto& audio_tensor = infer_result[0];
        const std::vector<int64_t> shape = audio_tensor.shape();
        auto* logits_data = static_cast<float*>(infer_result[0].data());
        const size_t element_count = audio_tensor.size();
        out_audio->insert(out_audio->end(), logits_data, logits_data + element_count);
        return true;
    }

    bool Kokoro::load_voices(const std::vector<std::string>& speaker_names,
                             std::vector<int64_t>& dims,
                             const std::string& voices_bin) {
        const size_t n_speaker = speaker_names.size();
        const int64_t max_len = style_dims_[0]; // 510
        const int64_t emb_dim = style_dims_[2]; // 256

        std::ifstream file(voices_bin, std::ios::binary);
        if (!file) {
            MD_LOG_ERROR << "fail to open " << voices_bin << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (n_speaker * max_len * emb_dim * sizeof(float) != file_size) {
            MD_LOG_ERROR << voices_bin << " file_size error, file size: " << file_size << " please check" << std::endl;
            return false;
        }

        // 2. 读取数据到 uint8 缓冲区
        std::vector<uint8_t> buffer(file_size);
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<int64_t>(file_size));
        file.close();

        // // 3. 重新解释为 float32 数组
        const auto float_data = reinterpret_cast<const float*>(buffer.data());
        for (int n = 0; n < n_speaker; ++n) {
            const size_t chunk_size = max_len * emb_dim;
            voices_[speaker_names[n]].assign(float_data + n * chunk_size,
                                             float_data + n * chunk_size + chunk_size);
        }
        return true;
    }

    void Kokoro::load_lexicons(const std::vector<std::string>& lexicon_files) {
        for (auto& fin : lexicon_files) {
            std::ifstream input(fin);
            std::string line;
            while (std::getline(input, line)) {
                auto arr = string_split(line, " ");
                const std::vector tokens(arr.begin() + 1, arr.end());
                word2token_[arr[0]] = tokens;
            }
        }
    }

    void Kokoro::load_tokens(const std::string& token_file) {
        std::ifstream input(token_file);
        std::string line;
        while (std::getline(input, line)) {
            auto arr = string_split(line, " ");
            if (arr.size() == 2) {
                token2id_[arr[0]] = stoi(arr[1]);
            }
            else {
                token2id_[" "] = stoi(arr[2]);
            }
        }
    }


    std::vector<std::string> Kokoro::split_ch_eng(const std::string& text) const {
        std::vector<std::string> ret;
        std::string cur;
        int cur_len = -1;
        for (size_t i = 0, len = 0; i < text.length(); i += len) {
            const unsigned char byte = static_cast<unsigned>(text[i]);
            if (byte >= 0xFC) // length 6
                len = 6;
            else if (byte >= 0xF8)
                len = 5;
            else if (byte >= 0xF0)
                len = 4;
            else if (byte >= 0xE0)
                len = 3;
            else if (byte >= 0xC0)
                len = 2;
            else
                len = 1;
            const auto sub = text.substr(i, len);
            const bool is_punc = punc_set_.find(text[i]) != punc_set_.end();
            const size_t tmp_len = is_punc ? 0 : len;
            if (cur_len != -1 && tmp_len != cur_len) {
                if (cur_len == 1) {
                    std::transform(cur.begin(), cur.end(), cur.begin(),
                                   [](unsigned char c) {
                                       return std::tolower(c);
                                   });
                }
                if (cur != " ") {
                    ret.push_back(cur);
                }
                cur = "";
            }
            cur += sub;
            cur_len = static_cast<int>(tmp_len);
        }
        if (!cur.empty()) {
            if (cur != " ") {
                ret.push_back(cur);
            }
        }
        return ret;
    }
}

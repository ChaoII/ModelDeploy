// csrc/audio/tts/qwen3/qwen3_tts_tokenizer.h
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace modeldeploy::audio::tts {

// Qwen3-TTS 12Hz 文本分词器（ByteLevel-BPE，读取 vocab.json + merges.txt）。
// 特殊 token（<|im_start|> 等）不在 vocab.json 中，由 tokenizer_config.json 的
// added_tokens_decoder + additional_special_tokens 补充并做整串最长匹配。
class Qwen3TtsTokenizer {
public:
    struct AddedToken {
        std::string content;
        int32_t id = -1;
        bool special = false;
    };

    struct TrieNode {
        std::unordered_map<uint8_t, int32_t> next;
        int32_t token_index = -1;
    };

    explicit Qwen3TtsTokenizer(const std::string& tokenizer_dir);
    Qwen3TtsTokenizer() = default;

    // 文本 -> token id。无法识别的字节级子串会被跳过。
    std::vector<int64_t> Encode(const std::string& text);

    [[nodiscard]] bool loaded() const { return !token2id_.empty(); }
    [[nodiscard]] int32_t vocab_size() const {
        return static_cast<int32_t>(token2id_.size());
    }

private:
    void BuildBytesToUnicode();
    void LoadTokenizerConfig(const std::string& config_blob);
    void Finalize();

    std::unordered_map<std::string, int32_t> token2id_;
    std::vector<std::string> id2token_;
    std::unordered_map<std::string, int32_t> merges_rank_;
    std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    std::vector<AddedToken> added_tokens_;
    std::vector<TrieNode> trie_;
    std::unordered_set<int32_t> special_ids_;
};

}  // namespace modeldeploy::audio::tts

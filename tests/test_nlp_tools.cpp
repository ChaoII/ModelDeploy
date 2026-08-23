#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include "nlp/tools/tokenizer.h"
namespace fs = std::filesystem;

static fs::path jieba_dir() {
    const char* dir = std::getenv("TEST_DATA_DIR");
    const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
    return base / "test_models" / "onnx" / "kokoro_v1_1" / "dict";
}

TEST_CASE("Tokenizer cuts a Chinese sentence", "[nlp]") {
    auto d = jieba_dir();
    if (!fs::exists(d / "jieba.dict.utf8")) { WARN("jieba 词典缺失（外链），跳过分词断言"); return; }
    modeldeploy::nlp::tool::Tokenizer t(d.string());
    REQUIRE(t.is_loaded());
    auto toks = t.tokenize("我爱北京天安门", "mix");
    REQUIRE_FALSE(toks.empty());
    REQUIRE(std::find(toks.begin(), toks.end(), "北京") != toks.end());
}
#include "nlp/tools/splitter.h"
#include "nlp/tools/normalizer.h"
#include "nlp/tools/keywords.h"
#include "nlp/tools/stats.h"

TEST_CASE("Splitter splits by punctuation", "[nlp]") {
    auto s = modeldeploy::nlp::tool::Splitter::split_sentences("你好。世界！你好吗？好的；行");
    REQUIRE(s.size() >= 5);
    REQUIRE(s[0] == "你好");
    REQUIRE(s[1] == "世界");
}

TEST_CASE("Normalizer normalizes digits", "[nlp]") {
    auto n = modeldeploy::nlp::tool::Normalizer::normalize("一共有１２３个苹果");
    REQUIRE(n.find("123") != std::string::npos);
}

TEST_CASE("Keywords top by frequency", "[nlp]") {
    auto kw = modeldeploy::nlp::tool::Keywords::top("apple apple banana apple", 2);
    REQUIRE(kw.size() == 2);
    REQUIRE(kw[0].first == "apple");
    REQUIRE(kw[0].second == 3);
}

TEST_CASE("Stats counts chars/words/sentences", "[nlp]") {
    using S = modeldeploy::nlp::tool::Stats;
    REQUIRE(S::word_count("hello world foo") == 3);
    REQUIRE(S::sentence_count("a。b！c") == 3);
    REQUIRE(S::char_count("你好") == 2);
}

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

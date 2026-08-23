#include "nlp/tools/tokenizer.h"
#if defined(BUILD_NLP)
#include <filesystem>
#include <cppjieba/Jieba.hpp>
namespace modeldeploy::nlp::tool {
struct Tokenizer::Impl { std::unique_ptr<cppjieba::Jieba> jieba; };

Tokenizer::Tokenizer(const std::string& dict_dir) : impl_(std::make_unique<Impl>()) {
    namespace fs = std::filesystem;
    const fs::path d(dict_dir);
    const auto dict    = (d / "jieba.dict.utf8").string();
    const auto hmm     = (d / "hmm_model.utf8").string();
    const auto user    = (d / "user.dict.utf8").string();
    const auto idf     = (d / "idf.utf8").string();
    const auto stop    = (d / "stop_words.utf8").string();
    if (!fs::exists(dict)) return;
    impl_->jieba = std::make_unique<cppjieba::Jieba>(dict, hmm, user, idf, stop);
    loaded_ = true;
}
Tokenizer::~Tokenizer() = default;
std::vector<std::string> Tokenizer::tokenize(const std::string& text, const std::string& mode) const {
    std::vector<std::string> out;
    if (!loaded_ || !impl_->jieba) return out;
    if (mode == "mp")           impl_->jieba->Cut(text, out, false);
    else if (mode == "hmm")     impl_->jieba->CutHMM(text, out);
    else if (mode == "full")    { std::vector<cppjieba::Word> ws; impl_->jieba->CutAll(text, ws); for (auto& w : ws) out.push_back(w.word); }
    else                        impl_->jieba->Cut(text, out, true); // mix（默认）
    return out;
}
} // namespace modeldeploy::nlp::tool
#endif // BUILD_NLP

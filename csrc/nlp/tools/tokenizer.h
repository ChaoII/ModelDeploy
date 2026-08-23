#pragma once
#include <memory>
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
class MODELDEPLOY_CXX_EXPORT Tokenizer {
public:
    explicit Tokenizer(const std::string& dict_dir);
    ~Tokenizer();
    bool is_loaded() const { return loaded_; }
    std::vector<std::string> tokenize(const std::string& text, const std::string& mode = "mix") const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_{false};
};
} // namespace modeldeploy::nlp::tool

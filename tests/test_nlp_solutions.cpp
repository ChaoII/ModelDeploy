#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "nlp/solutions/text_classifier.h"
using namespace modeldeploy;
using namespace modeldeploy::nlp::solution;

TEST_CASE("TextClassifier construction without weights", "[nlp]") {
    TextClassifier m("nonexistent_bert.onnx");
    REQUIRE_FALSE(m.is_initialized());
}

TEST_CASE("TextClassifier encode builds CLS/SEP ids + attention mask", "[nlp]") {
    std::vector<std::string> toks = {"我", "爱", "北京"};
    Tensor ids, mask;
    REQUIRE(TextClassifier::encode(toks, 101, 102, 0, 8, &ids, &mask));
    REQUIRE(ids.shape() == std::vector<int64_t>({1, 8}));
    const int32_t* idp = static_cast<const int32_t*>(ids.data());
    REQUIRE(idp[0] == 101);
    REQUIRE(idp[4] == 102);
    REQUIRE(idp[5] == 0);
}

TEST_CASE("TextClassifier softmax_top1 picks argmax", "[nlp]") {
    int label = -1; float score = 0;
    REQUIRE(TextClassifier::softmax_top1({1.0f, 5.0f, 2.0f}, &label, &score));
    REQUIRE(label == 1);
    REQUIRE(score > 0.9f);
    REQUIRE(score < 1.001f);
}

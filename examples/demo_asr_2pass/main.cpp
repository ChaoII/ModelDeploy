//
// Created by AC on 2025-01-13.
//
#include "src/asr/asr_2pass_capi.h"

int main() {
    //    MDModel model{strdup(""), {}, {}, nullptr};
    MDModel model{};
    md_create_two_pass_model(&model,
                             "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                             "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",
                             "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                             "D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
                             "D:/funasr-runtime-resources/models/fst_itn_zh",
                             "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst",
                             "",
                             ASRMode::TwoPass);
//    MDASRResult result{strdup(""), strdup(""), strdup(""), strdup(""), 0.0f};
    MDASRResult result{};
    md_two_pass_model_predict(&model, "D:/funasr-runtime-resources/vad_example.wav", &result);

    md_free_two_pass_result(&result);

    md_free_two_pass_model(&model);
}
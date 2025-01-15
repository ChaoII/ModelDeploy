//
// Created by aichao on 2025/1/15.
//

#include <src/asr/asr_capi.h>

#include "src/asr/asr_2pass_capi.h"
#include "src/asr/internal/audio.h"

void call_back(MDASRResult* result) {
    if (result->msg) {
        std::cout << "----" << result->msg << std::endl;
    }
    if (result->tpass_msg) {
        std::cout << "*****" << result->tpass_msg << std::endl;
    }

    md_free_asr_result(result);
}

int main() {
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

    const char* wav_path = "D:/funasr-runtime-resources/vad_example.wav";


    funasr::Audio audio(1);


    // PCM 数据的每个样本通常使用 16 位（2 字节）来表示

    int sampling_rate;
    audio.LoadWav2Char(wav_path, &sampling_rate);
    char* speech_buff = audio.GetSpeechChar();
    int buff_len = audio.GetSpeechLen() * 2;
    int step = 1000 * 2;
    bool is_final;

    for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
        if (sample_offset + step >= buff_len - 1) {
            step = buff_len - sample_offset;
            is_final = true;
        }
        else {
            is_final = false;
        }
        md_two_pass_model_predict_buffer(&model, speech_buff + sample_offset, step, is_final, "pcm", sampling_rate,
                                         call_back);
    }

    md_free_two_pass_model(&model);
}

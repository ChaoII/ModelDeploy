//
// Created by aichao on 2025/1/15.
//
#ifdef _WIN32
#include <Windows.h>
#endif

#include <chrono>
#include <src/asr/asr_offline_capi.h>
#include "src/asr/asr_2pass_capi.h"
#include "src/asr/internal/audio.h"
#include "src/log.h"

void call_back(MDASRResult* result) {
    if (result->msg) {
        MD_LOG_INFO("msg: {}", result->msg?result->msg:"");
    }
    if (result->stamp) {
        MD_LOG_INFO("stamp: {}", result->stamp);
    }
    if (result->stamp_sents) {
        MD_LOG_INFO("stamp_sents: {}", result->stamp_sents);
    }
    if (result->tpass_msg) {
        MD_LOG_INFO("tpass_msg: {}", result->tpass_msg);
    }

    md_free_asr_result(result);
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    MDModel model{};

    auto start_time1 = std::chrono::steady_clock::now();
    md_create_two_pass_model(&model,
                             "D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                             "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",
                             "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                             "D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx",
                             "D:/funasr-runtime-resources/models/fst_itn_zh",
                             "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst",
                             "",
                             ASRMode::TwoPass);
    auto end_time1 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    MD_LOG_INFO("md_create_asr_offline_model time: {}", duration);

    const char* wav_path = "D:/funasr-runtime-resources/vad_example.wav";
    funasr::Audio audio(1);
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
        md_two_pass_model_predict_buffer(&model, speech_buff + sample_offset, step, is_final,
                                         "pcm",
                                         sampling_rate,
                                         call_back);
    }
    md_free_two_pass_model(&model);
}

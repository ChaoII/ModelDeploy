//
// Created by aichao on 2026/8/10.
//

#include <cstring>
#include <string>
#include "csrc/audio/asr/sense_voice.h"
#include "csrc/core/md_log.h"
#include "csrc/utils/wave_helper.h"
#include "capi/common/md_micro.h"
#include "capi/audio/asr/sense_voice_capi.h"
#include "capi/utils/internal/utils.h"

MDStatusCode md_create_sense_voice_model(MDModel* model, const MDSenseVoiceParameters* asr_parameters,
                                         const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    if (asr_parameters == nullptr || asr_parameters->model_path == nullptr ||
        asr_parameters->tokens_path == nullptr) {
        return MDStatusCode::FileOpenFailed;
    }
    const auto asr_model = new modeldeploy::audio::asr::SenseVoice(
        asr_parameters->model_path, asr_parameters->tokens_path, _option);
    model->type = MDModelType::ASR;
    model->format = MDModelFormat::ONNX;
    model->model_content = asr_model;
    model->model_name = strdup(asr_model->name().c_str());
    if (!asr_model->is_initialized()) {
        MD_LOG_ERROR << "SenseVoice model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_sense_voice_model_predict(const MDModel* model, const char* wav_path,
                                          MDASRResult* c_result, int audio_fs) {
    if (model == nullptr || model->model_content == nullptr || wav_path == nullptr || c_result == nullptr) {
        return MDStatusCode::FileOpenFailed;
    }
    const auto asr_model = static_cast<modeldeploy::audio::asr::SenseVoice*>(model->model_content);
    std::vector<float> data;
    int32_t sample_rate = audio_fs;
    if (!load_wav_file(wav_path, &sample_rate, data)) {
        MD_LOG_ERROR << "Failed to load wav file: " << wav_path << std::endl;
        return MDStatusCode::FileOpenFailed;
    }
    std::string text;
    if (!asr_model->predict(data, &text)) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_result->msg = strdup(text.c_str());
    c_result->stamp = nullptr;
    c_result->stamp_sents = nullptr;
    c_result->tpass_msg = nullptr;
    c_result->snippet_time = 0.f;
    return MDStatusCode::Success;
}

void md_free_sense_voice_result(MDASRResult* c_result) {
    if (c_result == nullptr) return;
    if (c_result->msg != nullptr) {
        free(c_result->msg);
        c_result->msg = nullptr;
    }
    c_result->snippet_time = 0.f;
}

void md_free_sense_voice_model(MDModel* model) {
    if (model == nullptr) return;
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::audio::asr::SenseVoice*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}

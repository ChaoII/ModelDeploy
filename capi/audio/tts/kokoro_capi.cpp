//
// Created by AC on 2024/12/16.
//

#include <string>
#include <filesystem>
#include "csrc/audio/tts/kokoro.h"

#include "csrc/audio/tts/utils.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/audio/tts/kokoro_capi.h"


namespace fs = std::filesystem;

MDStatusCode md_create_kokoro_model(MDModel* model, const MDKokoroParameters* kokoro_parameters) {
    modeldeploy::RuntimeOption runtime_option;
    runtime_option.set_cpu_thread_num(kokoro_parameters->num_threads);
    const auto kokoro_model = new modeldeploy::audio::tts::Kokoro(
        fs::path(kokoro_parameters->model_path).string(),
        fs::path(kokoro_parameters->tokens_path).string(),
        {
            fs::path(kokoro_parameters->lexicons_en_path).string(),
            fs::path(kokoro_parameters->lexicons_zh_path).string()
        },
        fs::path(kokoro_parameters->voice_bin_path).string(),
        fs::path(kokoro_parameters->jieba_dir).string(),
        fs::path(kokoro_parameters->text_normalization_dir).string(),
        runtime_option);
    model->type = MDModelType::TTS;
    model->format = MDModelFormat::ONNX;
    model->model_content = kokoro_model;
    model->model_name = strdup(kokoro_model->name().c_str());
    if (!kokoro_model->is_initialized()) {
        MD_LOG_ERROR << "Kokoro model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_kokoro_model_predict(const MDModel* model, const char* input_text,
                                     const char* voice,
                                     const float speed, MDTTSResult* c_results) {
    std::vector<float> data;
    const auto kokoro_model = static_cast<modeldeploy::audio::tts::Kokoro*>(model->model_content);
    if (const bool res_status = kokoro_model->predict(input_text, voice, speed, &data); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_results->size = static_cast<int>(data.size());
    c_results->data = new float[data.size()];
    c_results->sample_rate = kokoro_model->get_sample_rate();
    std::ranges::copy(data, c_results->data);
    return MDStatusCode::Success;
}


MDStatusCode md_write_wav(const MDTTSResult* c_results, const char* output_path) {
    if (!modeldeploy::audio::tts::write_wave(output_path, c_results->sample_rate, c_results->data, c_results->size)) {
        return MDStatusCode::WriteWaveFailed;
    }
    return MDStatusCode::Success;
}

void md_free_kokoro_result(MDTTSResult* c_results) {
    if (c_results->size > 0 && c_results->data) {
        delete[] c_results->data;
    }
    c_results->data = nullptr;
    c_results->size = 0;
    c_results->sample_rate = 0;
}

void md_free_kokoro_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::audio::tts::Kokoro*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}

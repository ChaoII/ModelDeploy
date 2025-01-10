//
// Created by AC on 2025-01-07.
//
#include <fstream>
#include <sstream>
#include <filesystem>
#include "asr_capi.h"
#include "src/utils/internal/utils.h"
#include "internal/funasrruntime.h"
#include "internal/com-define.h"


namespace fs = std::filesystem;


struct ASRModel {
    void *asr_model;
    void *decoder_model;
    std::vector<std::vector<float>> hot_words_embedding;
};


MDStatusCode md_create_asr_model(MDModel *model,
                                 const char *model_dir,
                                 const char *vad_dir,
                                 const char *punct_dir,
                                 const char *itn_dir,
                                 const char *lm_dir,
                                 const char *hot_word_path,
                                 bool blade_disc,
                                 float global_beam,
                                 float lattice_beam,
                                 float am_scale,
                                 int fst_inc_wts,
                                 int thread_num,
                                 int batch_size,
                                 bool use_gpu) {


    std::map<std::string, std::string> model_path;
    model_path.insert({MODEL_DIR, model_dir});
    model_path.insert({QUANTIZE, is_quantize_model(model_dir) ? "true" : "false"});
    model_path.insert({BLADEDISC, blade_disc ? "true" : "false"});
    model_path.insert({VAD_DIR, vad_dir});
    model_path.insert({VAD_QUANT, is_quantize_model(vad_dir) ? "true" : "false"});
    model_path.insert({PUNC_DIR, punct_dir});
    model_path.insert({PUNC_QUANT, is_quantize_model(vad_dir) ? "true" : "false"});
    model_path.insert({ITN_DIR, itn_dir});
    model_path.insert({LM_DIR, lm_dir});
    void *asr_model = FunOfflineInit(model_path, thread_num, use_gpu, batch_size);
    if (!asr_model) {
        return MDStatusCode::ModelInitializeFailed;
    }
    float glob_beam = 3.0f;
    float lat_beam = 3.0f;
    float am_sc = 10.0f;
    if (lm_dir) {
        glob_beam = global_beam;
        lat_beam = lattice_beam;
        am_sc = am_scale;
    }
    // init wfst decoder
    void *decoder_handle = FunASRWfstDecoderInit(asr_model, ASR_OFFLINE, glob_beam, lat_beam, am_sc);
    // hotword file
    std::unordered_map<std::string, int> hws_map;
    std::string nn_hot_words;
    ExtractHotWords(hot_word_path, hws_map, nn_hot_words);
    // load hotwords list and build graph
    FunWfstDecoderLoadHwsRes(decoder_handle, fst_inc_wts, hws_map);
    std::vector<std::vector<float>> hot_words_embedding = CompileHotwordEmbedding(asr_model, nn_hot_words);
    auto model_content = new ASRModel{asr_model, decoder_handle, hot_words_embedding};
    model->type = MDModelType::ASR;
    model->format = MDModelFormat::ONNX;
    model->model_content = model_content;
    model->model_name = strdup("FunASR");
    return MDStatusCode::Success;
}

MDStatusCode md_asr_model_predict(MDModel *model, const char *wav_path, MDASRResult *asr_result, int audio_fs) {
    std::string wav_data;
    if (fs::path(wav_path).extension().string() == ".scp") {
        std::ifstream in(wav_path);
        if (!in.is_open()) {
            return MDStatusCode::FileOpenFailed;
        }
        std::string line;
        while (getline(in, line)) {
            std::istringstream iss(line);
            std::string column1, column2;
            iss >> column1 >> column2;
            wav_data = column2;
        }
        in.close();
    } else {
        wav_data = wav_path;
    }

    auto model_content = (ASRModel *) model->model_content;

    FUNASR_RESULT result = FunOfflineInfer(model_content->asr_model, wav_data.c_str(), RASR_NONE, nullptr,
                                           model_content->hot_words_embedding, audio_fs, true,
                                           model_content->decoder_model);
    if (!result) {
        return MDStatusCode::ModelPredictFailed;
    }
    std::string msg = FunASRGetResult(result, 0);
    std::string stamp = FunASRGetStamp(result);
    std::string stamp_sents = FunASRGetStampSents(result);
    std::string tpass_msg = FunASRGetTpassResult(result, 0);
    float snippet_time = FunASRGetRetSnippetTime(result);
    asr_result->msg = strdup(msg.c_str());
    asr_result->snippet_time = snippet_time;
    asr_result->stamp = strdup(stamp.c_str());
    asr_result->stamp_sents = strdup(stamp_sents.c_str());
    asr_result->tpass_msg = strdup(tpass_msg.c_str());
    FunASRFreeResult(result);
    return MDStatusCode::Success;
}

void md_free_asr_result(MDASRResult *asr_result) {

    if (asr_result->msg != nullptr) {
        free(asr_result->msg);
        asr_result->msg = nullptr;
    }
    if (asr_result->stamp != nullptr) {
        free(asr_result->stamp);
        asr_result->stamp = nullptr;
    }
    if (asr_result->stamp_sents != nullptr) {
        free(asr_result->stamp_sents);
        asr_result->stamp_sents = nullptr;
    }
    if (asr_result->tpass_msg != nullptr) {
        free(asr_result->tpass_msg);
        asr_result->tpass_msg = nullptr;
    }

}


void md_free_asr_model(MDModel *model) {
    if (model != nullptr) {
        if (model->model_content != nullptr) {
            auto model_content = static_cast<ASRModel *> (model->model_content);
            FunWfstDecoderUnloadHwsRes(model_content->decoder_model);
            FunASRWfstDecoderUninit(model_content->decoder_model);
            FunOfflineUninit(model_content->asr_model);
            delete model_content;
            model->model_content = nullptr;
        }
        if (model->model_name != nullptr) {
            free(model->model_name);
            model->model_name = nullptr;
        }
    }
}




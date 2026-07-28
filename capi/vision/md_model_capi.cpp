#include "capi/vision/md_model_capi.h"
#include "csrc/vision.h"

MDStatusCode md_clone_model(MDModel* model, const MDModel* from) {
    if (!model || !from || !from->model_content) {
        return MDStatusCode::CallError;
    }

    model->type = from->type;
    model->format = from->format;
    model->model_name = nullptr;

    auto* base = static_cast<modeldeploy::BaseModel*>(from->model_content);

    switch (from->type) {
        case MDModelType::Classification:
            model->model_content = dynamic_cast<modeldeploy::vision::classification::Classification*>(base)->clone().release();
            break;
        case MDModelType::Detection: {
            if (auto* p = dynamic_cast<modeldeploy::vision::detection::UltralyticsDet*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::detection::UltralyticsSeg*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::detection::UltralyticsObb*>(base)) {
                model->model_content = p->clone().release();
            } else {
                return MDStatusCode::ModelTypeError;
            }
            break;
        }
        case MDModelType::Keypoint:
            model->model_content = dynamic_cast<modeldeploy::vision::detection::UltralyticsPose*>(base)->clone().release();
            break;
        case MDModelType::OCR: {
            if (auto* p = dynamic_cast<modeldeploy::vision::ocr::PaddleOCR*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::ocr::PPStructureV2Table*>(base)) {
                model->model_content = p->clone().release();
            } else {
                return MDStatusCode::ModelTypeError;
            }
            break;
        }
        case MDModelType::FACE: {
            if (auto* p = dynamic_cast<modeldeploy::vision::face::Scrfd*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceID*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceAge*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceGender*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceAsFirst*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceAsSecond*>(base)) {
                model->model_content = p->clone().release();
            } else {
                return MDStatusCode::ModelTypeError;
            }
            break;
        }
        case MDModelType::LPR: {
            if (auto* p = dynamic_cast<modeldeploy::vision::lpr::LprDetection*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::lpr::LprRecognizer*>(base)) {
                model->model_content = p->clone().release();
            } else {
                return MDStatusCode::ModelTypeError;
            }
            break;
        }
        case MDModelType::PIPELINE: {
            if (auto* p = dynamic_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::FaceRecognizerPipeline*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::face::SeetaFaceAsPipeline*>(base)) {
                model->model_content = p->clone().release();
            } else if (auto* p = dynamic_cast<modeldeploy::vision::lpr::LprPipeline*>(base)) {
                model->model_content = p->clone().release();
            } else {
                return MDStatusCode::ModelTypeError;
            }
            break;
        }
        case MDModelType::ASR:
        case MDModelType::TTS:
            return MDStatusCode::ModelTypeError;
        default:
            return MDStatusCode::ModelTypeError;
    }

    if (!model->model_content) {
        return MDStatusCode::MemoryAllocatedFailed;
    }

    if (from->model_name) {
        model->model_name = strdup(from->model_name);
    }

    return MDStatusCode::Success;
}

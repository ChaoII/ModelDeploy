#include "capi/vision/md_model_capi.h"
#include "csrc/vision.h"

MDStatusCode md_clone_model(MDModel* model, const MDModel* from) {
    if (!model || !from || !from->model_content)
        return CallError;

    model->type = from->type;
    model->format = from->format;
    model->model_name = nullptr;

    switch (from->type) {
    case Detection:
        model->model_content = static_cast<modeldeploy::vision::detection::UltralyticsDet*>(from->model_content)->clone().release();
        break;
    case InstanceSeg:
        model->model_content = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(from->model_content)->clone().release();
        break;
    case OBB:
        model->model_content = static_cast<modeldeploy::vision::detection::UltralyticsObb*>(from->model_content)->clone().release();
        break;
    case Keypoint:
        model->model_content = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(from->model_content)->clone().release();
        break;
    case Classification:
        model->model_content = static_cast<modeldeploy::vision::classification::Classification*>(from->model_content)->clone().release();
        break;
    case FaceDet:
        model->model_content = static_cast<modeldeploy::vision::face::Scrfd*>(from->model_content)->clone().release();
        break;
    case FaceRec:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceID*>(from->model_content)->clone().release();
        break;
    case FaceAge:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceAge*>(from->model_content)->clone().release();
        break;
    case FaceGender:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceGender*>(from->model_content)->clone().release();
        break;
    case FaceASFirst:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceAsFirst*>(from->model_content)->clone().release();
        break;
    case FaceASSecond:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceAsSecond*>(from->model_content)->clone().release();
        break;
    case FaceASPipeline:
        model->model_content = static_cast<modeldeploy::vision::face::SeetaFaceAsPipeline*>(from->model_content)->clone().release();
        break;
    case FaceRecPipeline:
        model->model_content = static_cast<modeldeploy::vision::face::FaceRecognizerPipeline*>(from->model_content)->clone().release();
        break;
    case LPR:
        model->model_content = static_cast<modeldeploy::vision::lpr::LprDetection*>(from->model_content)->clone().release();
        break;
    case LPRRec:
        model->model_content = static_cast<modeldeploy::vision::lpr::LprRecognizer*>(from->model_content)->clone().release();
        break;
    case LPRPipeline:
        model->model_content = static_cast<modeldeploy::vision::lpr::LprPipeline*>(from->model_content)->clone().release();
        break;
    case OCRPipeline:
        model->model_content = static_cast<modeldeploy::vision::ocr::PaddleOCR*>(from->model_content)->clone().release();
        break;
    case OCRRec:
        model->model_content = static_cast<modeldeploy::vision::ocr::Recognizer*>(from->model_content)->clone().release();
        break;
    case OCR:
        model->model_content = static_cast<modeldeploy::vision::ocr::PPStructureV2Table*>(from->model_content)->clone().release();
        break;
    case PIPELINE:
        model->model_content = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(from->model_content)->clone().release();
        break;
    default:
        return ModelTypeError;
    }

    if (!model->model_content)
        return MemoryAllocatedFailed;
    return Success;
}

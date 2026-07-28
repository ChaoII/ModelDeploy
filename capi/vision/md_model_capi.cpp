#include "capi/vision/md_model_capi.h"
#include "csrc/base_model.h"

MDStatusCode md_clone_model(MDModel* model, const MDModel* from) {
    if (!model || !from || !from->model_content) {
        return CallError;
    }

    auto* base = static_cast<modeldeploy::BaseModel*>(from->model_content);
    auto cloned = base->clone_base();
    if (!cloned) {
        return ModelTypeError;
    }

    model->type = from->type;
    model->format = from->format;
    model->model_name = nullptr;
    model->model_content = cloned.release();
    return Success;
}

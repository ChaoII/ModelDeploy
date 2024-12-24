//
// Created by AC on 2024-12-17.
//

#pragma once


#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif


EXPORT_DECL StatusCode create_detection_model(WModel *model, const char *model_dir, int thread_num = 8,
                                              ModelFormat format = ModelFormat::ONNX);

EXPORT_DECL StatusCode set_detection_input_size(WModel *model, WSize size);

EXPORT_DECL StatusCode detection_predict(WModel *model, WDetectionResults *results, WImage *image,
                                         int draw_result, WColor color, double alpha, int is_save_result);

EXPORT_DECL void print_detection_result(WDetectionResults *result);

EXPORT_DECL void free_detection_result(WDetectionResults *result);

EXPORT_DECL void free_detection_model(WModel *model);

#ifdef __cplusplus
}
#endif

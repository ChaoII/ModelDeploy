//
// Created by AC on 2024-12-17.
//

#pragma once


#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void *DetectionModelHandle;


EXPORT_DECL DetectionModelHandle create_detection_model(const char *model_dir, int thread_num = 8);

EXPORT_DECL void set_detection_input_size(DetectionModelHandle model, WSize size);

EXPORT_DECL WDetectionResult* detection_predict(DetectionModelHandle model, WImage *image,
                                               int draw_result, WColor color, double alpha, int is_save_result);

EXPORT_DECL void print_detection_result(WDetectionResult *result);

EXPORT_DECL void free_detection_result(WDetectionResult *result);

#ifdef __cplusplus
}
#endif

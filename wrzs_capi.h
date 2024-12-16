//
// Created by AC on 2024/12/16.
//
#pragma once

#include "decl.h"


#ifdef __cplusplus
extern "C" {
#endif
typedef void *OCRModelHandle;

enum WImageType {
    WImageType_BGR,
    WImageType_GRAY
};

enum StatusCode {
    Success = 0x00,
    ModelInitFailed = 0x01,
    ModelPredictFailed = 0x02,
};

typedef struct {
    int x;
    int y;
} WPoint;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} WRect;


typedef struct {
    int width;
    int height;
    unsigned char *data;
    WImageType type;
} WImage;

typedef struct {
    WPoint *points;
    size_t size;
} WPolygon;


typedef struct {
    WPolygon *boxes;
    char **texts;
    float *scores;
    size_t size;
} WOCRResult;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} WColor;



/// 初始化模型
/// \param model_dir 模型所在目录（最好使用posix的格式，即单个正斜杠'c:/xxx/xxx'），目录最后不要带斜杠
/// \param dict_file 文本识别字典文件
/// \param thread_num 增加线程数，效率并不会线性增加，默认为1
EXPORT_DECL OCRModelHandle create_ocr_model(const char *model_dir, const char *dict_file, int thread_num = 1);

EXPORT_DECL StatusCode text_rec_buffer(OCRModelHandle model, WImage *image, WOCRResult *out_data,
                                       int draw_result, WColor color, double alpha, int is_save_result);

EXPORT_DECL WRect get_text_position(OCRModelHandle model, WImage *image, const char *text);

EXPORT_DECL WRect get_template_position(WImage *shot_img, WImage *template_img);

EXPORT_DECL WPoint get_center_point(WRect rect);

EXPORT_DECL void free_wimage(WImage *img);

EXPORT_DECL WImage *read_image(const char *path);

EXPORT_DECL void free_ocr_result(WOCRResult *result);

EXPORT_DECL void free_ocr_model(OCRModelHandle model);

#ifdef __cplusplus
}
#endif

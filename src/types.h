//
// Created by AC on 2024-12-17.
//

#pragma once

enum MDModelFormat {
    ONNX = 0,
    PaddlePaddle = 1,
};

enum MDModelType {
    OCR = 0,
    Detection = 1,
};

enum MDStatusCode {
    Success = 0x00,
    CallError = 0x01,
    ModelInitializeFailed = 0x03,
    ModelPredictFailed = 0x04,
    MemoryAllocatedFailed = 0x05,
    OCRDetModelInitializeFailed = 0x06,
    OCRRecModelInitializeFailed = 0x07,
};

typedef struct {
    char *model_name;
    MDModelType type;
    MDModelFormat format;
    void *model_content;
} MDModel;

typedef struct {
    int x;
    int y;
} MDPoint;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} MDRect;


typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} MDImage;

typedef struct {
    MDPoint *data;
    size_t size;
} MDPolygon;

typedef struct {
    /// ocr model directory;
    const char *model_dir;
    /// ocr dictionary path
    const char *dict_path;
    /// thread num default is 8
    int thread_num;
    /// model format default is PaddlePaddle
    MDModelFormat format;
    /// maximum side length default 960
    int max_side_len;
    /// db threshold default 0.3
    double det_db_thresh;
    /// db box threshold default 0.6
    double det_db_box_thresh;
    /// detect db unclip ratio default 1.5
    double det_db_unclip_ratio;
    /// detect db score mode default is "slow"
    const char *det_db_score_mode;
    /// is use dilation default is false(0)
    int use_dilation;
    /// recognition batch size default is 8, unusually set the same as thread_num
    int rec_batch_size;
} MDOCRModelParameters;

typedef struct {
    MDPolygon box;
    char *text;
    float score;
} MDOCRResult;

typedef struct {
    MDOCRResult *data;
    size_t size;
} MDOCRResults;


typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} MDColor;

typedef struct {
    int width;
    int height;
} MDSize;

typedef struct {
    MDRect box;
    int label_id;
    float score;
} MDDetectionResult;


typedef struct {
    MDDetectionResult *data;
    size_t size;
} MDDetectionResults;



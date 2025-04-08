//
// Created by AC on 2024-12-17.
//

#pragma once

enum MDASRMode {
    Offline = 0,
    Online = 1,
    TwoPass = 2,
};


enum MDModelFormat {
    ONNX = 0,
    PaddlePaddle = 1,
    Tennis = 2,
};

enum MDModelType {
    OCR = 0,
    Detection = 1,
    FACE = 2,
    ASR = 3,
    TTS = 4
};

enum MDStatusCode {
    Success = 0x00,
    PathNotFound,
    FileOpenFailed,
    CallError,
    ModelInitializeFailed,
    ModelPredictFailed,
    MemoryAllocatedFailed,
    OCRDetModelInitializeFailed,
    OCRRecModelInitializeFailed,
    ModelTypeError,
};

typedef struct {
    char* model_name;
    MDModelType type;
    MDModelFormat format;
    void* model_content;
} MDModel;

typedef struct {
    int x;
    int y;
} MDPoint;

typedef struct {
    double x;
    double y;
} MDPointF;

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
    unsigned char* data;
} MDImage;

typedef struct {
    MDPoint* data;
    int size;
} MDPolygon;

typedef struct {
    /// ocr model directory;
    const char* model_dir;
    /// ocr dictionary path
    const char* dict_path;
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
    const char* det_db_score_mode;
    /// is use dilation default is false(0)
    int use_dilation;
    /// recognition batch size default is 8, unusually set the same as thread_num
    int rec_batch_size;
} MDOCRModelParameters;

typedef struct {
    const char* model_path;
    int use_itn;
    const char* language;
    const char* tokens_path;
    int num_threads;
    int debug;
} MDSenseVoiceParameters;

typedef struct {
    const char* model_path;
    const char* voices_path;
    const char* tokens_path;
    const char* data_dir;
    const char* dict_dir;
    const char* lexicon;
    int num_threads;
    int debug;
} MDKokoroParameters;


typedef struct {
    MDPolygon box;
    char* text;
    float score;
} MDOCRResult;

typedef struct {
    MDOCRResult* data;
    int size;
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
    MDRect box;
    MDPoint* landmarks;
    int landmarks_size;
    int label_id;
    float score;
} MDDetectionLandmarkResult;


typedef struct {
    MDDetectionResult* data;
    int size;
} MDDetectionResults;

typedef struct {
    MDDetectionLandmarkResult* data;
    int size;
} MDDetectionLandmarkResults;


typedef struct {
    float* embedding;
    int size;
} MDFaceRecognizerResult;

typedef struct {
    MDFaceRecognizerResult* data;
    int size;
} MDFaceRecognizerResults;


typedef int MDFaceAgeResult;

typedef struct {
    int label_id;
    float score;
} MDFaceAsSecondResult;

typedef struct {
    MDFaceAsSecondResult* data;
    int size;
} MDFaceAsSecondResults;

enum MDFaceGenderResult {
    FEMALE = 0,
    MALE = 1
};

enum MDFaceAsResult {
    REAL = 0,
    FUZZY = 1,
    SPOOF = 2
};

typedef struct {
    MDFaceAsResult* data;
    int size;
} MDFaceAsResults;


enum MDFaceQualityEvaluateType {
    BRIGHTNESS = 0,
    CLARITY = 1,
    INTEGRITY = 2,
    POSE = 3,
    RESOLUTION = 4,
    CLARITY_EX = 5,
    NO_MASK = 6
};

enum MDFaceQualityEvaluateResult {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2
};


enum MDEyeState {
    EYE_CLOSE = 0,
    EYE_OPEN = 1,
    EYE_RANDOM = 2,
    EYE_UNKNOWN = 3
};

typedef struct {
    MDEyeState left_eye;
    MDEyeState right_eye;
} MDEyeStateResult;


typedef struct {
    char* msg;
    char* stamp;
    char* stamp_sents;
    char* tpass_msg;
    float snippet_time;
} MDASRResult;

typedef void (*ASRCallBack)(MDASRResult* result);

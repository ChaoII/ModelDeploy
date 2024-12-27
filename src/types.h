//
// Created by AC on 2024-12-17.
//

#pragma once

#define MD_FACE_DETECT             (0b00000001)
#define MD_FACE_LANDMARK           (0b00000001 << 1)
#define MD_FACE_RECOGNITION        (0b00000001 << 2)
#define MD_FACE_ANTI_SPOOfING      (0b00000001 << 3)
#define MD_FACE_QUALITY_EVALUATE   (0b00000001 << 4)
#define MD_FACE_AGE_ATTRIBUTE      (0b00000001 << 5)
#define MD_FACE_GENDER_ATTRIBUTE   (0b00000001 << 6)
#define MD_FACE_EYE_STATE          (0b00000001 << 7)


#define MD_MASK (MD_FACE_DETECT | MD_FACE_LANDMARK | MD_FACE_RECOGNITION|MD_FACE_AGE_ATTRIBUTE|MD_FACE_GENDER_ATTRIBUTE|MD_FACE_EYE_STATE)


enum MDModelFormat {
    ONNX = 0,
    PaddlePaddle = 1,
    Tennis = 2,
};

enum MDModelType {
    OCR = 0,
    Detection = 1,
    FACE = 2,
};

enum MDStatusCode {
    Success = 0x00,
    CallError = 0x01,
    ModelInitializeFailed = 0x02,
    ModelPredictFailed = 0x03,
    MemoryAllocatedFailed = 0x04,
    OCRDetModelInitializeFailed = 0x05,
    OCRRecModelInitializeFailed = 0x06,

    ModelTypeError = 0x07,
    NotFoundLandmark = 0x08,
    NotFoundFace = 0x09,
    FaceFeatureExtractError,

    FaceDetectionFlagNotSetError,
    FaceLandmarkFlagNotSetError,
    FaceRecognitionFlagNotSetError,
    FaceAntiSpoofingFlagNotSetError,
    FaceQualityEvaluateFlagNotSetError,
    FaceAgeAttributeFlagNotSetError,
    FaceGenderAttributeFlagNotSetError,
    FaceEyeStateFlagNotSetError,
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

typedef struct {
    MDPointF *data;
    size_t size;
} MDLandMarkResult;

typedef struct {
    float *data;
    size_t size;
} MDFaceFeature;


enum MDFaceAntiSpoofingResult {
    REAL = 0,
    SPOOF = 1,
    FUZZY = 2,
    DETECTING = 3
};

enum MDFaceQualityEvaluateType {
    BRIGHTNESS = 0,
    CLARITY = 1,
    INTEGRITY = 2,
    POSE = 3,
    RESOLUTION = 4,
    CLEAR = 5,
    NO_MASK = 6
};

enum MDFACEQualityEvaluateResult {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2
};

enum MDGenderResult {
    MALE = 0,
    FEMALE = 1
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



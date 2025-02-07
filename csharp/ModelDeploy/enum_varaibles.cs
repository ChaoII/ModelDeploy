namespace ModelDeploy;

public enum MDASRMode
{
    Offline = 0,
    Online = 1,
    TwoPass = 2,
};

public enum MDModelFormat
{
    ONNX = 0,
    PaddlePaddle = 1,
    Tennis = 2,
};

public enum MDModelType
{
    OCR = 0,
    Detection = 1,
    FACE = 2,
    ASR = 3,
};

public enum MDStatusCode
{
    Success = 0x00,
    FileOpenFailed = 0x01,
    CallError,
    ModelInitializeFailed,
    ModelPredictFailed,
    MemoryAllocatedFailed,
    OCRDetModelInitializeFailed,
    OCRRecModelInitializeFailed,

    ModelTypeError,
    NotFoundLandmark,
    NotFoundFace,
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

public enum FaceQualityEvaluateType
{
    Brightness = 0,
    Clarity = 1,
    Integrity = 2,
    Pose = 3,
    Resolution = 4,
    ClarityEx = 5,
    NoMask = 6
}

public enum MDEyeState
{
    EYE_CLOSE = 0,
    EYE_OPEN = 1,
    EYE_RANDOM = 2,
    EYE_UNKNOWN = 3
}
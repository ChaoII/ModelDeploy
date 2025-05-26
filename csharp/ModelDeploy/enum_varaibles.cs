namespace ModelDeploy
{
    public enum MDModelFormat
    {
        ONNX = 0,
        MNN
    }

    public enum MDModelType
    {
        Classification,
        Detection = 1,
        OCR,
        FACE,
        LPR,
        ASR,
        TTS
    }

    public enum MDStatusCode
    {
        Success = 0x00,
        PathNotFound,
        FileOpenFailed,
        CallError,
        ModelInitializeFailed,
        ModelPredictFailed,
        MemoryAllocatedFailed,
        ModelTypeError,
        WriteWaveFailed
    }

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
}
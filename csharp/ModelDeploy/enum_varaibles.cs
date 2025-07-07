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

    public enum Device
    {
        CPU = 0,
        GPU = 1,
        OPENCL = 2,
        VULKAN = 3
    }

    public enum Backend
    {
        ORT = 0,
        MNN = 1,
        TRT = 2,
        NONE = 3
    }
}
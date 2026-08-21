namespace ModelDeploy
{
    /// <summary>设备枚举（与 capi MD_DEVICE 对齐）。</summary>
    public enum Device
    {
        CPU = 0,
        GPU = 1,
        TPU = 2,
        OPENCL = 3,
        VULKAN = 4
    }

    /// <summary>后端枚举（与 capi MD_BACKEND 对齐）。</summary>
    public enum Backend
    {
        ORT = 0,
        MNN = 1,
        TRT = 2,
        SOPHGO = 3
    }
}

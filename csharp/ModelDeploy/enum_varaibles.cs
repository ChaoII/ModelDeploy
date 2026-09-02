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

    /// <summary>视频编解码后端（对应 capi MDCodecBackend）。</summary>
    public enum VideoCodecBackend
    {
        Auto = 0,
        FFmpeg = 1,
        GStreamer = 2
    }

    /// <summary>视频硬件加速（对应 capi MDHwAccel）。</summary>
    public enum VideoHwAccel
    {
        Auto = 0,
        None = 1,
        Cuda = 2,
        Vaapi = 3,
        Sophgo = 4,
        Qsv = 5
    }

    /// <summary>异步队列满时的背压策略（对应 capi MDBackpressure）。</summary>
    public enum VideoBackpressure
    {
        Block = 0,
        Drop = 1,
        OverwriteOldest = 2
    }

    /// <summary>会话状态（对应 capi MDVideoState）。</summary>
    public enum VideoState
    {
        Idle = 0,
        Opening = 1,
        Running = 2,
        Reconnecting = 3,
        Eof = 4,
        Error = 5,
        Closed = 6
    }
}

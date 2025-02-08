using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.audio.asr;

public class ASRResult
{
    public required string Message { get; set; }
    public required string Stamp { get; set; }
    public required string StampSents { get; set; }
    public required string TwoPassMessage { get; set; }
    public float SnippetTime { get; set; }

    public static ASRResult FromMDASRResult(MDASRResult cresult)
    {
        ASRResult result = new ASRResult
        {
            Message = Marshal.PtrToStringUTF8(cresult.msg) ?? "",
            Stamp = Marshal.PtrToStringUTF8(cresult.stamp) ?? "",
            StampSents = Marshal.PtrToStringUTF8(cresult.stamp_sents) ?? "",
            TwoPassMessage = Marshal.PtrToStringUTF8(cresult.tpass_msg) ?? "",
            SnippetTime = cresult.snippet_time
        };
        return result;
    }
};
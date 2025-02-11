using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.audio.asr
{
    public class ASRResult
    {
        public  string Message { get; set; }
        public  string Stamp { get; set; }
        public  string StampSents { get; set; }
        public  string TwoPassMessage { get; set; }
        public float SnippetTime { get; set; }

        public static ASRResult FromMDASRResult(MDASRResult cresult)
        {
            ASRResult result = new ASRResult
            {
                Message = Utils.PtrToStringUTF8(cresult.msg) ?? "",
                Stamp = Utils.PtrToStringUTF8(cresult.stamp) ?? "",
                StampSents = Utils.PtrToStringUTF8(cresult.stamp_sents) ?? "",
                TwoPassMessage = Utils.PtrToStringUTF8(cresult.tpass_msg) ?? "",
                SnippetTime = cresult.snippet_time
            };
            return result;
        }
    };
}

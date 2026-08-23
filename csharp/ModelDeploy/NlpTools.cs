using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy
{
    /// <summary>NLP 纯工具（对应 CAPI md_nlp_split_sent / md_nlp_stats，无权重）。</summary>
    public static class NlpTools
    {
        public static string[] SplitSentences(string text)
        {
            var tp = Utf8.Alloc(text);
            try
            {
                if (md_nlp_split_sent(tp, out var sents, out var n) != MDStatus.MD_OK) return Array.Empty<string>();
                var count = (int)n;
                var result = new string[count];
                var ptrs = new IntPtr[count];
                Marshal.Copy(sents, ptrs, 0, count);
                for (var i = 0; i < count; i++)
                    result[i] = ResultReader.ReadString(ptrs[i]) ?? string.Empty;
                return result;
            }
            finally { Utf8.Free(tp); }
        }

        public static (ulong Chars, ulong Words, ulong Sents) Stats(string text)
        {
            var tp = Utf8.Alloc(text);
            try
            {
                if (md_nlp_stats(tp, out var chars, out var words, out var sents) != MDStatus.MD_OK)
                    return (0, 0, 0);
                return (chars.ToUInt64(), words.ToUInt64(), sents.ToUInt64());
            }
            finally { Utf8.Free(tp); }
        }
    }

    /// <summary>NLP 文本分类器（对应 CAPI MD_MODEL_TEXT_CLASSIFIER / md_nlp_classify）。</summary>
    public sealed class NlpClassifier : BaseModel
    {
        private NlpClassifier(IntPtr handle)
            : base(MDModelKind.MD_MODEL_TEXT_CLASSIFIER, handle) { }

        public NlpClassifier(string modelPath, RuntimeOption opt = null)
            : base(MDModelKind.MD_MODEL_TEXT_CLASSIFIER, modelPath, opt) { }

        public (int Label, float Score) Predict(string text)
        {
            var tp = Utf8.Alloc(text);
            try
            {
                if (md_nlp_classify(_handle, tp, out var label, out var score) != MDStatus.MD_OK)
                    throw new InvalidOperationException("classify failed");
                return (label, score);
            }
            finally { Utf8.Free(tp); }
        }
    }
}

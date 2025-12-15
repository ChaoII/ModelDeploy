using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.pipeline
{
    public class AttributeResult
    {
        private Rect Box { get; set; }
        private int BoxLabelId { get; set; }
        private float BoxScore { get; set; }
        private List<float> AttrScores { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDAttributeResult>();

        private static AttributeResult FromNative(MDAttributeResult cResult)
        {
            var attrResult = new AttributeResult();
            attrResult.Box = Rect.FromNative(cResult.box);
            attrResult.BoxLabelId = cResult.box_label_id;
            attrResult.BoxScore = cResult.box_score;
            float[] buffer = new float[cResult.attr_scores_size];
            Marshal.Copy(cResult.attr_scores, buffer, 0, cResult.attr_scores_size);
            attrResult.AttrScores = new List<float>(buffer);
            return attrResult;
        }

        private static MDAttributeResult ToNative(AttributeResult result)
        {
            var cResult = new MDAttributeResult();
            cResult.box = result.Box.ToNative();
            cResult.box_label_id = result.BoxLabelId;
            cResult.box_score = result.BoxScore;
            cResult.attr_scores_size = result.AttrScores.Count;
            // 此处手动开辟内存会在C语言中释放
            cResult.attr_scores = Marshal.AllocHGlobal(cResult.attr_scores_size * sizeof(float));
            Marshal.Copy(result.AttrScores.ToArray(), 0, cResult.attr_scores, cResult.attr_scores_size);
            return cResult;
        }

        public static List<AttributeResult> FromNativeArray(MDAttributeResults cResults)
        {
            var results = new List<AttributeResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDAttributeResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDAttributeResults ToNativeArray(IReadOnlyList<AttributeResult> results)
        {
            var cResults = new MDAttributeResults();

            cResults.size = results.Count;
            cResults.data = results.Count > 0
                ? Marshal.AllocHGlobal(results.Count * NativeSize)
                : IntPtr.Zero;
            for (int i = 0; i < results.Count; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = ToNative(results[i]);
                Marshal.StructureToPtr(native, ptr, false);
            }

            return cResults;
        }
    }
}
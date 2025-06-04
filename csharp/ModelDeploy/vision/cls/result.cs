using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.classification
{
    public class ClassificationResult
    {
        private int LabelId { get; set; }
        private float Score { get; set; }

        public override string ToString() => $"LabelId: {LabelId}, Score: {Score}";

        private static readonly int NativeSize = Marshal.SizeOf<MDClassificationResult>();

        private static ClassificationResult FromNative(MDClassificationResult cResult) => new ClassificationResult
        {
            LabelId = cResult.label_id,
            Score = cResult.score
        };

        private static MDClassificationResult ToNative(ClassificationResult result) => new MDClassificationResult
        {
            label_id = result.LabelId,
            score = result.Score
        };

        public static List<ClassificationResult> FromNativeArray(MDClassificationResults cResults)
        {
            var results = new List<ClassificationResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDClassificationResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDClassificationResults ToNativeArray(List<ClassificationResult> results)
        {
            var cResults = new MDClassificationResults
            {
                size = results.Count,
                data = results.Count > 0
                    ? Marshal.AllocHGlobal(results.Count * NativeSize)
                    : IntPtr.Zero
            };

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
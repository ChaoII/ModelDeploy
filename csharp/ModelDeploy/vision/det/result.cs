using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.detection
{
    public class DetectionResult
    {
        private Rect Box { get; set; }
        private int LabelId { get; set; }
        private float Score { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDDetectionResult>();

        private static DetectionResult FromNative(MDDetectionResult cResult) => new DetectionResult
        {
            Box = Rect.FromNative(cResult.box),
            LabelId = cResult.label_id,
            Score = cResult.score
        };

        private static MDDetectionResult ToNative(DetectionResult result) => new MDDetectionResult
        {
            box = result.Box.ToNative(),
            label_id = result.LabelId,
            score = result.Score
        };

        public static List<DetectionResult> FromNativeArray(MDDetectionResults cResults)
        {
            var results = new List<DetectionResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDDetectionResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDDetectionResults ToNativeArray(IReadOnlyList<DetectionResult> results)
        {
            var cResults = new MDDetectionResults
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
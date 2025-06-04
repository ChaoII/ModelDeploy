using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.obb
{
    public class ObbResult
    {
        private RotatedRect Box { get; set; }
        private int LabelId { get; set; }
        private float Score { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDObbResult>();

        private static ObbResult FromNative(MDObbResult cResult) => new ObbResult
        {
            Box = RotatedRect.FromNative(cResult.box),
            LabelId = cResult.label_id,
            Score = cResult.score
        };

        private static MDObbResult ToNative(ObbResult result) => new MDObbResult
        {
            box = result.Box.ToNative(),
            label_id = result.LabelId,
            score = result.Score
        };

        public static List<ObbResult> FromNativeArray(MDObbResults cResults)
        {
            var results = new List<ObbResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDObbResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDObbResults ToNativeArray(IReadOnlyList<ObbResult> results)
        {
            var cResults = new MDObbResults
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
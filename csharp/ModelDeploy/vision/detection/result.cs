using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.detection
{
    public class Mask
    {
        private List<byte> Buffer { get; set; }
        private List<int> Shape { get; set; }

        public static Mask FromNative(MDMask cResult)
        {
            var mask = new Mask
            {
                Buffer = new List<byte>(cResult.buffer_size),
                Shape = new List<int>(cResult.num_dims)
            };
            if (cResult.buffer_size <= 0 || cResult.num_dims <= 0) return mask;
            byte[] buffer = new byte[cResult.buffer_size];
            Marshal.Copy(cResult.buffer, buffer, 0, cResult.buffer_size);
            mask.Buffer = new List<byte>(buffer);
            int[] shape = new int[cResult.num_dims];
            Marshal.Copy(cResult.shape, shape, 0, cResult.num_dims);
            mask.Shape = new List<int>(shape);
            return mask;
        }

        public static MDMask ToNative(Mask result)
        {
            if (result.Buffer.Count == 0 || result.Shape.Count == 0)
            {
                return new MDMask
                {
                    buffer = IntPtr.Zero,
                    buffer_size = result.Buffer.Count,
                    num_dims = result.Shape.Count,
                    shape = IntPtr.Zero
                };
            }

            var cResult = new MDMask
            {
                buffer = Marshal.AllocHGlobal(result.Buffer.Count * sizeof(byte)),
                buffer_size = result.Buffer.Count,
                num_dims = result.Shape.Count,
                shape = Marshal.AllocHGlobal(result.Shape.Count * sizeof(int))
            };

            byte[] buffer = result.Buffer.ToArray();
            Marshal.Copy(buffer, 0, cResult.buffer, buffer.Length);

            int[] shape = result.Shape.ToArray();
            Marshal.Copy(shape, 0, cResult.shape, shape.Length);
            return cResult;
        }
    }


    public class DetectionResult
    {
        private Rect Box { get; set; }
        private Mask Mask { get; set; }
        private int LabelId { get; set; }
        private float Score { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDDetectionResult>();

        private static DetectionResult FromNative(MDDetectionResult cResult) => new DetectionResult
        {
            Box = Rect.FromNative(cResult.box),
            Mask = Mask.FromNative(cResult.mask),
            LabelId = cResult.label_id,
            Score = cResult.score
        };

        private static MDDetectionResult ToNative(DetectionResult result) => new MDDetectionResult
        {
            box = result.Box.ToNative(),
            mask = Mask.ToNative(result.Mask),
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
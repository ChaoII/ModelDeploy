using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.sem
{
    public class SemSegResult
    {
        public List<byte> Labels { get; set; }
        public List<int> Shape { get; set; }
        public int NumClasses { get; set; }

        public static SemSegResult FromNative(MDSemSegResult cResult)
        {
            var result = new SemSegResult
            {
                NumClasses = cResult.num_classes,
                Shape = new List<int>(cResult.shape_size)
            };
            if (cResult.shape_size > 0 && cResult.shape != IntPtr.Zero)
            {
                int[] shape = new int[cResult.shape_size];
                Marshal.Copy(cResult.shape, shape, 0, cResult.shape_size);
                result.Shape = new List<int>(shape);

                int labelCount = 1;
                for (int i = 0; i < shape.Length; i++)
                    labelCount *= shape[i];

                if (labelCount > 0 && cResult.labels != IntPtr.Zero)
                {
                    byte[] labels = new byte[labelCount];
                    Marshal.Copy(cResult.labels, labels, 0, labelCount);
                    result.Labels = new List<byte>(labels);
                }
            }
            return result;
        }

        public static MDSemSegResult ToNative(SemSegResult result)
        {
            var cResult = new MDSemSegResult
            {
                num_classes = result.NumClasses,
                shape_size = result.Shape.Count,
                shape = Marshal.AllocHGlobal(result.Shape.Count * sizeof(int)),
                labels = result.Labels != null && result.Labels.Count > 0
                    ? Marshal.AllocHGlobal(result.Labels.Count * sizeof(byte))
                    : IntPtr.Zero
            };
            if (result.Shape.Count > 0)
                Marshal.Copy(result.Shape.ToArray(), 0, cResult.shape, result.Shape.Count);
            if (result.Labels != null && result.Labels.Count > 0)
                Marshal.Copy(result.Labels.ToArray(), 0, cResult.labels, result.Labels.Count);
            return cResult;
        }

        public static void FreeNative(ref MDSemSegResult cResult)
        {
            if (cResult.shape != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(cResult.shape);
                cResult.shape = IntPtr.Zero;
            }
            if (cResult.labels != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(cResult.labels);
                cResult.labels = IntPtr.Zero;
            }
            cResult.shape_size = 0;
            cResult.num_classes = 0;
        }
    }
}

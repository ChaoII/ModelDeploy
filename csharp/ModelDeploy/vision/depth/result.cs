using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.depth
{
    public class DepthResult
    {
        public List<float> Depth { get; set; }
        public List<int> Shape { get; set; }

        public static DepthResult FromNative(MDDepthResult cResult)
        {
            var result = new DepthResult
            {
                Shape = new List<int>(cResult.shape_size)
            };
            if (cResult.shape_size > 0 && cResult.shape != IntPtr.Zero)
            {
                int[] shape = new int[cResult.shape_size];
                Marshal.Copy(cResult.shape, shape, 0, cResult.shape_size);
                result.Shape = new List<int>(shape);

                int depthCount = 1;
                for (int i = 0; i < shape.Length; i++)
                    depthCount *= shape[i];

                if (depthCount > 0 && cResult.depth != IntPtr.Zero)
                {
                    float[] depth = new float[depthCount];
                    Marshal.Copy(cResult.depth, depth, 0, depthCount);
                    result.Depth = new List<float>(depth);
                }
            }
            return result;
        }

        public static MDDepthResult ToNative(DepthResult result)
        {
            var cResult = new MDDepthResult
            {
                shape_size = result.Shape.Count,
                shape = Marshal.AllocHGlobal(result.Shape.Count * sizeof(int)),
                depth = result.Depth != null && result.Depth.Count > 0
                    ? Marshal.AllocHGlobal(result.Depth.Count * sizeof(float))
                    : IntPtr.Zero
            };
            if (result.Shape.Count > 0)
                Marshal.Copy(result.Shape.ToArray(), 0, cResult.shape, result.Shape.Count);
            if (result.Depth != null && result.Depth.Count > 0)
                Marshal.Copy(result.Depth.ToArray(), 0, cResult.depth, result.Depth.Count);
            return cResult;
        }

        public static void FreeNative(ref MDDepthResult cResult)
        {
            if (cResult.shape != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(cResult.shape);
                cResult.shape = IntPtr.Zero;
            }
            if (cResult.depth != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(cResult.depth);
                cResult.depth = IntPtr.Zero;
            }
            cResult.shape_size = 0;
        }
    }
}

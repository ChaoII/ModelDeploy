using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.pose
{
    public class PoseResult
    {
        public Rect Box { get; set; }
        public List<Point3f> Keypoints { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }


        private static readonly int NativeSize = Marshal.SizeOf<MDPoseResult>();

        private static PoseResult FromNative(MDPoseResult cResult)
        {
            var poseResult = new PoseResult
            {
                Box = Rect.FromNative(cResult.box),
                LabelId = cResult.label_id,
                Score = cResult.score,
                Keypoints = new List<Point3f>(cResult.keypoints_size),
            };
            for (var i = 0; i < cResult.keypoints_size; i++)
            {
                var currentPtr = IntPtr.Add(cResult.keypoints_data, i * Marshal.SizeOf<MDPoint3f>());
                var cPoint = Marshal.PtrToStructure<MDPoint3f>(currentPtr);
                poseResult.Keypoints.Add(Point3f.FromNative(cPoint));
            }

            return poseResult;
        }

        private static MDPoseResult ToNative(PoseResult result)
        {
            var cResult = new MDPoseResult()
            {
                box = result.Box.ToNative(),
                label_id = result.LabelId,
                score = result.Score,
                keypoints_size = result.Keypoints.Count,
                keypoints_data = result.Keypoints.Count > 0
                    ? Marshal.AllocHGlobal(result.Keypoints.Count * Marshal.SizeOf<MDPoint3f>())
                    : IntPtr.Zero,
            };
            for (var i = 0; i < result.Keypoints.Count; i++)
            {
                var currentPtr = IntPtr.Add(cResult.keypoints_data, i * Marshal.SizeOf<MDPoint3f>());
                Marshal.StructureToPtr(result.Keypoints[i].ToNative(), currentPtr, false);
            }

            return cResult;
        }

        public static List<PoseResult> FromNativeArray(MDPoseResults cResults)
        {
            var results = new List<PoseResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDPoseResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDPoseResults ToNativeArray(List<PoseResult> results)
        {
            var cResults = new MDPoseResults
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
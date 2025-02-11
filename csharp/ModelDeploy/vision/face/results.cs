using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.face
{


    //MDFaceAntiSpoofingResult
    public enum FaceAntiSpoofingResult
    {
        Real = 0,
        Spoof = 1,
        Fuzzy = 2,
        Detecting = 3
    }

    //MDFaceQualityEvaluateResult
    public enum FaceQualityEvaluateResult
    {
        Low = 0,
        Medium = 1,
        High = 2
    }

    //MDGenderResult
    public enum GenderResult
    {
        Male = 0,
        Female = 1
    }

    public class EyeStateResult
    {
        public MDEyeState LeftEye { get; set; }
        public MDEyeState RightEye { get; set; }

        public override string ToString()
        {
            return $"LeftEye: {LeftEye}, RightEye: {RightEye}";
        }

        public static EyeStateResult FromRow(MDEyeStateResult cResult)
        {
            return new EyeStateResult
            {
                LeftEye = cResult.left_eye,
                RightEye = cResult.right_eye
            };
        }
    }

    public class LandMarkResult
    {

        public LandMarkResult(List<PointF> points)
        {
            Points = points;
        }

        private List<PointF> Points { get; set; } = null;

        public static LandMarkResult FromRow(MDLandMarkResult cResult)
        {
            List<PointF> landMarkPoints = new List<PointF>();
            for (int i = 0; i < cResult.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResult.data, i * Marshal.SizeOf<MDPointF>());
                MDPointF point = Marshal.PtrToStructure<MDPointF>(currentPtr);
                landMarkPoints.Add(PointF.FromRaw(point));
            }

            return new LandMarkResult(landMarkPoints);
        }

        public static MDLandMarkResult ToRow(LandMarkResult result)
        {
            MDLandMarkResult cResult = new MDLandMarkResult
            {
                // notice that the data will destroy in C
                data = Marshal.AllocHGlobal(result.Points.Count * Marshal.SizeOf<MDPointF>()),
                size = result.Points.Count
            };
            for (int i = 0; i < result.Points.Count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResult.data, i * Marshal.SizeOf<MDPointF>());
                MDPointF res = PointF.ToRaw(result.Points[i]);
                Marshal.StructureToPtr(res, currentPtr, false);
            }

            return cResult;
        }
    }

    public class FaceFeature
    {
        FaceFeature(List<float> feature)
        {
            Feature  = feature;
        }

        private List<float> Feature { get; set; }


        public static FaceFeature FromRow(MDFaceFeature cResult)
        {
            List<float> result = new List<float>();
            for (int i = 0; i < cResult.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResult.data, i * Marshal.SizeOf<float>());
                float res = Marshal.PtrToStructure<float>(currentPtr);
                result.Add(res);
            }

            return new FaceFeature(result);
        }

        public static MDFaceFeature ToRow(FaceFeature result)
        {
            MDFaceFeature cFeature = new MDFaceFeature
            {
                // notice that the data will destroy in C
                data = Marshal.AllocHGlobal(result.Feature.Count * Marshal.SizeOf<float>()),
                size = result.Feature.Count
            };
            for (int i = 0; i < result.Feature.Count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cFeature.data, i * Marshal.SizeOf<float>());
                float res = result.Feature[i];
                Marshal.StructureToPtr(res, currentPtr, false);
            }

            return cFeature;
        }
    }
}
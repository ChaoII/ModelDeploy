using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.face
{
    public class InsightFaceResult
    {
        public Rect Box { get; set; }
        public float Score { get; set; }
        public List<Point3F> Kps { get; set; } = new List<Point3F>();
        public List<Point3F> Landmark2D106 { get; set; } = new List<Point3F>();
        public List<Point3F> Landmark3D68 { get; set; } = new List<Point3F>();
        public float[] Pose { get; set; } = new float[3];
        public float[] Embedding { get; set; } = new float[0];

        internal static InsightFaceResult FromNative(MDInsightFaceResult c)
        {
            var r = new InsightFaceResult
            {
                Box = Rect.FromNative(c.box),
                Score = c.score,
                Pose = c.pose ?? new float[3],
            };
            r.Kps = ReadPoints(c.kps, c.kps_size);
            r.Landmark2D106 = ReadPoints(c.landmark_2d_106, c.landmark_2d_106_size);
            r.Landmark3D68 = ReadPoints(c.landmark_3d_68, c.landmark_3d_68_size);
            r.Embedding = ReadFloats(c.embedding, c.embedding_size);
            return r;
        }

        internal static List<InsightFaceResult> FromNativeArray(MDInsightFaceResults cResults)
        {
            var list = new List<InsightFaceResult>(cResults.size);
            var size = Marshal.SizeOf<MDInsightFaceResult>();
            for (var i = 0; i < cResults.size; i++)
            {
                var ptr = IntPtr.Add(cResults.data, i * size);
                var c = Marshal.PtrToStructure<MDInsightFaceResult>(ptr);
                list.Add(FromNative(c));
            }
            return list;
        }

        private static List<Point3F> ReadPoints(IntPtr ptr, int count)
        {
            var list = new List<Point3F>(count);
            var size = Marshal.SizeOf<MDPoint3f>();
            for (var i = 0; i < count; i++)
            {
                var p = Marshal.PtrToStructure<MDPoint3f>(IntPtr.Add(ptr, i * size));
                list.Add(Point3F.FromNative(p));
            }
            return list;
        }

        private static float[] ReadFloats(IntPtr ptr, int count)
        {
            if (ptr == IntPtr.Zero || count <= 0) return Array.Empty<float>();
            var arr = new float[count];
            Marshal.Copy(ptr, arr, 0, count);
            return arr;
        }
    }

    /// <summary>insightface 综合人脸分析（det + 2D/3D landmark + recognition）</summary>
    public sealed class InsightFaceAnalysis : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        internal InsightFaceAnalysis(MDModel existing)
        {
            _model = existing;
        }

        public InsightFaceAnalysis(string detModelPath, string recModelPath,
            string lmk2dModelPath, string lmk3dModelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(
                md_create_insightface_model(ref _model, detModelPath, recModelPath,
                    lmk2dModelPath, lmk3dModelPath, ref nativeOption),
                "Create insightface model");
        }

        public List<InsightFaceResult> Analyze(Image image)
        {
            var cResults = new MDInsightFaceResults();
            Utils.Check(md_insightface_analyze(ref _model, ref image.RawImage, ref cResults),
                "Insightface analyze");
            try
            {
                return InsightFaceResult.FromNativeArray(cResults);
            }
            finally
            {
                md_free_insightface_result(ref cResults);
            }
        }

        public void SetDetThresh(float thresh)
        {
            md_insightface_set_det_thresh(ref _model, thresh);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_insightface_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~InsightFaceAnalysis() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_insightface_model(ref MDModel model,
            string detModelPath, string recModelPath, string lmk2dModelPath, string lmk3dModelPath,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_insightface_analyze(ref MDModel model, ref MDImage image,
            ref MDInsightFaceResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_insightface_set_det_thresh(ref MDModel model, float thresh);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_insightface_result(ref MDInsightFaceResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_insightface_model(ref MDModel model);

        #endregion
    }
}

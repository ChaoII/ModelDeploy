using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.pose
{
    public sealed class UltralyticsPose : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public UltralyticsPose(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_pose_model(ref _model, modelDir, threadNum), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_pose_input_size(ref _model, size), "Set detection input size");
        }

        public List<PoseResult> Predict(Image image)
        {
            var cResults = new MDPoseResults();
            Utils.Check(md_pose_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<PoseResult>(PoseResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_pose_result(ref cResults);
            }
        }


        public void Display(List<PoseResult> results)
        {
            var cResults = PoseResult.ToNativeArray(results);
            try
            {
                md_print_pose_result(ref cResults);
            }
            finally
            {
                md_free_pose_result(ref cResults);
            }
        }


        public void DrawPoseResult(Image image, List<PoseResult> results, string fontPath, int fontSize = 12,
            int keypointRadius = 4, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = PoseResult.ToNativeArray(results);
            try
            {
                md_draw_pose_result(ref image.RawImage, ref cResults, fontPath, fontSize,
                    keypointRadius, alpha, saveResult);
            }
            finally
            {
                md_free_pose_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_pose_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsPose() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_pose_model(ref MDModel model, string modelDir, int threadNum);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_pose_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_pose_predict(ref MDModel model, ref MDImage image,
            ref MDPoseResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_pose_result(ref MDPoseResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_pose_result(ref MDImage image, ref MDPoseResults result,
            string fontPath, int fontSize, int keypointRadius, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_pose_result(ref MDPoseResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_pose_model(ref MDModel model);

        #endregion
    }
}
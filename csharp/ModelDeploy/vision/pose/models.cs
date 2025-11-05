using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;
using ModelDeploy.vision.face;

namespace ModelDeploy.vision.pose
{
    public sealed class UltralyticsPose : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public UltralyticsPose(string modelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_keypoint_model(ref _model, modelPath, ref nativeOption), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_keypoint_input_size(ref _model, size), "Set detection input size");
        }

        public List<KeyPointResult> Predict(Image image)
        {
            var cResults = new MDKeyPointResults();
            Utils.Check(md_keypoint_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<KeyPointResult>(KeyPointResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }


        public void Display(List<KeyPointResult> results)
        {
            var cResults = KeyPointResult.ToNativeArray(results);
            try
            {
                md_print_keypoint_result(ref cResults);
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }


        public void DrawPoseResult(Image image, List<KeyPointResult> results, string fontPath, int fontSize = 12,
            int keypointRadius = 4, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = KeyPointResult.ToNativeArray(results);
            try
            {
                md_draw_keypoint_result(ref image.RawImage, ref cResults, fontPath, fontSize,
                    keypointRadius, alpha, saveResult);
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_keypoint_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsPose() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_keypoint_model(ref MDModel model, string modelPath, ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_keypoint_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_keypoint_predict(ref MDModel model, ref MDImage image,
            ref MDKeyPointResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_keypoint_result(ref MDKeyPointResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_keypoint_result(ref MDImage image, ref MDKeyPointResults result,
            string fontPath, int fontSize, int keypointRadius, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_keypoint_result(ref MDKeyPointResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_keypoint_model(ref MDModel model);

        #endregion
    }
}
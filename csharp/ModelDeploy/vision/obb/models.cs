using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.obb
{
    public sealed class UltralyticsObb : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public UltralyticsObb(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_obb_model(ref _model, modelDir, threadNum), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_obb_input_size(ref _model, size), "Set detection input size");
        }

        public List<ObbResult> Predict(Image image)
        {
            var cResults = new MDObbResults();
            Utils.Check(md_obb_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<ObbResult>(ObbResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_obb_result(ref cResults);
            }
        }


        public void Display(List<ObbResult> results)
        {
            var cResults = ObbResult.ToNativeArray(results);
            try
            {
                md_print_obb_result(ref cResults);
            }
            finally
            {
                md_free_obb_result(ref cResults);
            }
        }


        public void DrawObbResult(Image image, List<ObbResult> results,
            double threshold, string fontPath, int fontSize = 12, double alpha = 0.5, int saveResult = 0)
        {
            var cResults = ObbResult.ToNativeArray(results);
            try
            {
                md_draw_obb_result(ref image.RawImage, ref cResults, threshold, fontPath, fontSize, alpha, saveResult);
            }
            finally
            {
                md_free_obb_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_obb_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsObb() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_obb_model(ref MDModel model, string modelDir, int threadNum);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_obb_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_obb_predict(ref MDModel model, ref MDImage image,
            ref MDObbResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_obb_result(ref MDObbResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_obb_result(ref MDImage image, ref MDObbResults result,
            double threshold, string fontPath, int fontSize, double alpha, int saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_obb_result(ref MDObbResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_obb_model(ref MDModel model);

        #endregion
    }
}
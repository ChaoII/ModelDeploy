using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.iseg
{
    public sealed class UltralyticsSeg : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public UltralyticsSeg(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_instance_seg_model(ref _model, modelDir, threadNum), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_instance_seg_input_size(ref _model, size), "Set detection input size");
        }

        public List<ISegResult> Predict(Image image)
        {
            var cResults = new MDISegResults();
            Utils.Check(md_instance_seg_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<ISegResult>(ISegResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_instance_seg_result(ref cResults);
            }
        }

        public void Display(List<ISegResult> results)
        {
            var cResults = ISegResult.ToNativeArray(results);
            try
            {
                md_print_instance_seg_result(ref cResults);
            }
            finally
            {
                md_print_instance_seg_result(ref cResults);
            }
        }

        public void DrawIsegResult(Image image, List<ISegResult> results, double threshold,
            string fontPath, int fontSize = 12, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = ISegResult.ToNativeArray(results);
            try
            {
                md_draw_instance_seg_result(ref image.RawImage, ref cResults, threshold, fontPath, fontSize, alpha,
                    saveResult);
            }
            finally
            {
                md_free_instance_seg_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_instance_seg_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsSeg() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_instance_seg_model(ref MDModel model, string modelDir, int threadNum);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_instance_seg_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_instance_seg_predict(ref MDModel model, ref MDImage image,
            ref MDISegResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_instance_seg_result(ref MDISegResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_instance_seg_result(ref MDImage image, ref MDISegResults result,
            double threshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_instance_seg_result(ref MDISegResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_instance_seg_model(ref MDModel model);

        #endregion
    }
}
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.instance_seg
{
    public sealed class YoloV5Seg : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public YoloV5Seg(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_instance_seg_model(ref _model, modelDir, threadNum), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_instance_seg_input_size(ref _model, size), "Set detection input size");
        }

        public List<detection.DetectionResult> Predict(Image image)
        {
            var cResults = new MDDetectionResults();
            Utils.Check(md_instance_seg_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<detection.DetectionResult>(detection.DetectionResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_instance_seg_result(ref cResults);
            }
        }

        public void Display(List<detection.DetectionResult> results)
        {
            var cResults = detection.DetectionResult.ToNativeArray(results);
            try
            {
                md_print_instance_seg_result(ref cResults);
            }
            finally
            {
                md_print_instance_seg_result(ref cResults);
            }
        }

        public void DrawDetectionResult(Image image, List<detection.DetectionResult> results, string fontPath,
            int fontSize = 12, double alpha = 0.5, int saveResult = 0)
        {
            var cResults = detection.DetectionResult.ToNativeArray(results);
            try
            {
                md_draw_instance_seg_result(ref image.RawImage, ref cResults, fontPath, fontSize, alpha, saveResult);
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

        ~YoloV5Seg() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_instance_seg_model(ref MDModel model, string modelDir, int threadNum);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_instance_seg_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_instance_seg_predict(ref MDModel model, ref MDImage image,
            ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_instance_seg_result(ref MDDetectionResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_instance_seg_result(ref MDImage image, ref MDDetectionResults result,
            string fontPath, int fontSize, double alpha, int saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_instance_seg_result(ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_instance_seg_model(ref MDModel model);

        #endregion
    }
}
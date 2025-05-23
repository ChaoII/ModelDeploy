using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.detection
{
    public sealed class YoloV8 : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public YoloV8(string modelDir, int threadNum = 8, MDModelFormat format = MDModelFormat.ONNX)
        {
            _model = new MDModel();
            Utils.Check(md_create_detection_model(ref _model, modelDir, threadNum, format), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_detection_input_size(ref _model, size), "Set detection input size");
        }

        public List<DetectionResult> Predict(Image image)
        {
            var cResults = new MDDetectionResults();
            Utils.Check(md_detection_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<DetectionResult>(DetectionResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_detection_result(ref cResults);
            }
        }

        public void DrawDetectionResult(Image image, List<DetectionResult> results, string fontPath, int fontSize = 12,
            double alpha = 0.5, int saveResult = 1)
        {
            var cResults = DetectionResult.ToNativeArray(results);
            try
            {
                md_draw_detection_result(ref image.RawImage, ref cResults, fontPath, fontSize, alpha, saveResult);
            }
            finally
            {
                md_free_detection_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_detection_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~YoloV8() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_detection_model(ref MDModel model, string modelDir, int threadNum,
            MDModelFormat format);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_detection_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_detection_predict(ref MDModel model, ref MDImage image,
            ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_detection_result(ref MDImage image, ref MDDetectionResults result,
            string fontPath, int fontSize, double alpha, int saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_free_detection_result(ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_detection_model(ref MDModel model);

        #endregion
    }
}
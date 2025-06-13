using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.classification
{
    public sealed class UltralyticsCls : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public UltralyticsCls(string modelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_classification_model(ref _model, modelPath, ref nativeOption),
                "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_classification_input_size(ref _model, size), "Set detection input size");
        }

        public List<ClassificationResult> Predict(Image image)
        {
            var cResults = new MDClassificationResults();
            Utils.Check(md_classification_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<ClassificationResult>(ClassificationResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_classification_result(ref cResults);
            }
        }

        public void DrawClassificationResult(
            Image image, List<ClassificationResult> results,
            string fontPath, int topK, float scoreThreshold, int fontSize = 12,
            double alpha = 0.5, bool saveResult = false)
        {
            var cResults = ClassificationResult.ToNativeArray(results);
            try
            {
                md_draw_classification_result(ref image.RawImage, ref cResults, topK, scoreThreshold, fontPath,
                    fontSize, alpha, saveResult);
            }
            finally
            {
                md_free_classification_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_classification_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsCls() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_classification_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_classification_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_classification_predict(ref MDModel model, ref MDImage image,
            ref MDClassificationResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_classification_result(ref MDImage image, ref MDClassificationResults result,
            int topK, float scoreThreshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_classification_result(ref MDClassificationResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_classification_model(ref MDModel model);

        #endregion
    }
}
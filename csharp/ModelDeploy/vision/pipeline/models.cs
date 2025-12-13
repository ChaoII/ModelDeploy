using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.pipeline
{
    public sealed class  PedestrianAttribute: IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public PedestrianAttribute(string detModelOath,string clsModelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeRuntimeOption = option.ToNative();
            Utils.Check(md_create_detection_model(ref _model, modelDir, ref nativeRuntimeOption),
                "Create PedestrianAttribute model");
        }

        public void SetDetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_detection_input_size(ref _model, size), "Set detection input size");
        }
        public void SetClsInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_detection_input_size(ref _model, size), "Set detection input size");
        }
        
        public void SetClsBatchSize(int batchSize)
        {
            Utils.Check(md_set_detection_input_size(ref _model, size), "Set detection input size");
        }
        
                
        public void SetDetThreshold(float threshold)
        {
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


        public void Display(List<DetectionResult> results)
        {
            var cResults = DetectionResult.ToNativeArray(results);
            try
            {
                md_print_detection_result(ref cResults);
            }
            finally
            {
                md_free_detection_result(ref cResults);
            }
        }


        public void DrawDetectionResult(Image image, List<DetectionResult> results,
            double threshold, string fontPath, int fontSize = 12, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = DetectionResult.ToNativeArray(results);
            try
            {
                md_draw_detection_result(ref image.RawImage, ref cResults,
                    threshold, fontPath, fontSize, alpha, saveResult);
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

        ~UltralyticsDet() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_detection_model(ref MDModel model, string modelDir,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_detection_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_detection_predict(ref MDModel model, ref MDImage image,
            ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_detection_result(ref MDDetectionResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_detection_result(ref MDImage image, ref MDDetectionResults result,
            double threshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_detection_result(ref MDDetectionResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_detection_model(ref MDModel model);

        #endregion
    }
}
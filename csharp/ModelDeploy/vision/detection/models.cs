using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.detection
{

    public class YOLOv8
    {
        private MDModel _model;

        public YOLOv8(string modelDir, int threadNum = 8, MDModelFormat format = MDModelFormat.ONNX)
        {
            _model = new MDModel();
            int ret = md_create_detection_model(ref _model, modelDir, threadNum, format);
            if (ret != 0)
            {
                throw new Exception("md_create_detection_model failed error code is: " + ret);
            }
        }

        ~YOLOv8()
        {
            md_free_detection_model(ref _model);
        }

        public void SetInputSize(int width, int height)
        {
            MDSize size = new MDSize { width = width, height = height };
            int ret = md_set_detection_input_size(ref _model, size);
            if (ret != 0)
            {
                throw new Exception("md_set_detection_input_size failed error code is: " + ret);
            }
        }

        public List<DetectionResult> Predict(Image image)
        {
            MDDetectionResults cResults = new MDDetectionResults();
            int ret = md_detection_predict(ref _model, ref image.RawImage, ref cResults);
            if (ret != 0)
            {
                throw new Exception("md_detection_predict failed error code is: " + ret);
            }

            List<DetectionResult> detectionResult = DetectionResult.FromMDDetectionResults(cResults);
            md_free_detection_result(ref cResults);
            return detectionResult;
        }

        public void DrawDetectionResult(Image image, List<DetectionResult> results, string fontPath, int fontSize = 12,
            double alpha = 0.5, int saveResult = 1)
        {
            MDDetectionResults cResults = DetectionResult.ToMDDetectionResults(results);
            md_draw_detection_result(ref image.RawImage, ref cResults, fontPath, fontSize, alpha, saveResult);
            md_free_detection_result(ref cResults);
        }

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_detection_model(ref MDModel model, string modelDir, int threadNum = 8,
            MDModelFormat format = MDModelFormat.ONNX);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_detection_input_size(ref MDModel model, MDSize size);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int
            md_detection_predict(ref MDModel model, ref MDImage image, ref MDDetectionResults results);


        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_detection_result(ref MDImage image, ref MDDetectionResults result,
            string fontPath, int fontSize, double alpha, int saveResult);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_free_detection_result(ref MDDetectionResults results);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_detection_model(ref MDModel model);
    }
}

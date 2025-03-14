using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.ocr
{
    public class PPOCRv4
    {
        private MDModel model;

        public PPOCRv4(MDOCRModelParameters parameters)
        {
            model = new MDModel();
            int ret = md_create_ocr_model(ref model, ref parameters);
            if (ret != 0)
            {
                throw new Exception("md_create_ocr_model failed error code is: " + ret);
            }
        }
        ~PPOCRv4()
        {
            md_free_ocr_model(ref model);
        }

        public List<OCRResult> Predict(Image image)
        {
            MDOCRResults cResults = new MDOCRResults();
            int ret = md_ocr_model_predict(ref model, ref image.RawImage, ref cResults);
            if (ret != 0)
            {
                throw new Exception("md_ocr_model_predict failed error code is: " + ret);
            }
            List<OCRResult> ocrResult = OCRResult.FromMDOCRResults(cResults);
            md_free_ocr_result(ref cResults);
            return ocrResult;
        }

        public Rect GetTextPosition(Image image, string text)
        {
            MDRect rect = md_get_text_position(ref model, ref image.RawImage, text);
            return Rect.FromRaw(rect);
        }

        public void DrawOcrResult(Image image, List<OCRResult> results, string fontPath, Color color, int fontSize = 12,
            double alpha = 0.5, int saveResult = 1)
        {
            MDOCRResults cResults = OCRResult.ToMDOCRResults(results);
            MDColor cColor = color.ToRaw();
            md_draw_ocr_result(ref image.RawImage, ref cResults, fontPath, fontSize, ref cColor, alpha, 1);
            md_free_ocr_result(ref cResults);
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_ocr_model(ref MDModel model, ref MDOCRModelParameters parameters);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_model_predict(ref MDModel model, ref MDImage image, ref MDOCRResults results);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDRect md_get_text_position(ref MDModel model, ref MDImage image, string text);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_ocr_result(ref MDImage image, ref MDOCRResults results, string fontPath,
            int fontSize, ref MDColor color, double alpha, int saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_result(ref MDOCRResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_model(ref MDModel model);
    }

    public class OCRRecognition
    {
        private MDModel _model;
        
        public OCRRecognition(string modelPath, string dictPath, int threadNum = 1)
        {
            _model = new MDModel();
            int ret = md_create_ocr_recognition_model(ref _model, modelPath, dictPath, threadNum);
            if (ret != 0)
            {
                md_free_ocr_recognition_model(ref _model);
                throw new Exception("md_create_ocr_model failed error code is: " + ret);
            }
        }

        ~OCRRecognition()
        {
            md_free_ocr_recognition_model(ref _model);
        }

        public OCRResult Predict(Image image)
        {
            MDOCRResult cResult = new MDOCRResult();
            int ret = md_ocr_recognition_model_predict(ref _model, ref image.RawImage, ref cResult);
            if (ret != 0)
            {
                throw new Exception("md_ocr_model_predict failed error code is: " + ret);
            }

            OCRResult ocrResult = OCRResult.FromMDOCRResult(cResult);
            md_free_ocr_recognition_result(ref cResult);
            return ocrResult;
        }


        public List<OCRResult> BatchPredict(Image image, List<Polygon> polygons, int batchSize)
        {
            MDOCRResults cResults = new MDOCRResults();
            MDPolygon[] cPolygons = null;
            try
            {
                cPolygons = Polygon.ToMDPolygonArray(polygons);
                int ret = md_ocr_recognition_model_predict_batch(ref _model, ref image.RawImage, batchSize,
                    ref cPolygons[0], polygons.Count, ref cResults);
                if (ret != 0)
                {
                    throw new Exception("md_ocr_model_predict failed error code is: " + ret);
                }
                return OCRResult.FromMDOCRResults(cResults);
            }
            finally
            {
                if (cPolygons != null)
                {
                    foreach (var polygon in cPolygons)
                    {
                        Polygon.FreeMDPolygon(polygon);
                    }
                }
                md_free_ocr_result(ref cResults);
            }
        }


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int
            md_create_ocr_recognition_model(ref MDModel model, string modelPath, string dictPath, int threadNum = 1);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_recognition_model_predict(ref MDModel model, ref MDImage image,
            ref MDOCRResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_recognition_model_predict_batch(ref MDModel model, ref MDImage image,
            int batchSize, ref MDPolygon polygon, int size, ref MDOCRResults results);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_recognition_result(ref MDOCRResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_recognition_model(ref MDModel model);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_result(ref MDOCRResults results);
    }
}
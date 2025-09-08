using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.ocr
{
    public class PaddleOcr : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public PaddleOcr(MDOCRModelParameters parameters, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            if (md_create_ocr_model(ref _model, ref parameters, ref nativeOption) != 0)
                throw new Exception("md_create_ocr_model failed.");
        }

        public void SetMaxSideLen(int maxSideLen)
        {
            Utils.Check(md_ocr_det_set_max_side_len(ref _model, maxSideLen), "ocr_det_set_max_side_len");
        }

        public void UseDilation(bool useDilation = false)
        {
            Utils.Check(md_ocr_det_db_set_use_dilation(ref _model, useDilation ? 1 : 0), "ocr_det_db_set_use_dilation");
        }

        public void UnclipRatio(double dbUnclipRatio = 1.5)
        {
            Utils.Check(md_ocr_det_db_unclip_ratio(ref _model, dbUnclipRatio), "ocr_det_db_unclip_ratio");
        }

        public void SetDbThresh(double dbThresh = 0.3)
        {
            Utils.Check(md_ocr_det_set_db_thresh(ref _model, dbThresh), "ocr_det_set_db_thresh");
        }

        public void SetDbBoxThresh(double dbBoxThresh = 0.6)
        {
            Utils.Check(md_ocr_det_set_db_box_thresh(ref _model, dbBoxThresh), "ocr_det_set_db_box_thresh");
        }

        public OcrResults Predict(Image image)
        {
            MDOCRResults cResults = new MDOCRResults();
            Utils.Check(md_ocr_model_predict(ref _model, ref image.RawImage, ref cResults), "md_ocr_model_predict");
            var results = OcrResults.FromNativeArray(cResults);
            md_free_ocr_result(ref cResults);
            return results;
        }

        public Rect GetTextPosition(Image image, string text)
        {
            var nativeRect = md_get_text_position(ref _model, ref image.RawImage, text);
            return Rect.FromNative(nativeRect);
        }

        public void DrawOcrResult(Image image, OcrResults results, string fontPath, int fontSize = 12,
            double alpha = 0.5, bool saveResult = false)
        {
            var nativeResults = OcrResults.ToNativeArray(results);
            md_draw_ocr_result(ref image.RawImage, ref nativeResults, fontPath, fontSize, alpha, saveResult);
            md_free_ocr_result(ref nativeResults);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_ocr_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~PaddleOcr() => Dispose();

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_ocr_model(ref MDModel model, ref MDOCRModelParameters parameters,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_det_set_max_side_len(ref MDModel model, int maxSideLen = 960);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_det_set_db_thresh(ref MDModel model, double dbThresh = 0.3);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_det_set_db_box_thresh(ref MDModel model, double dbBoxThresh = 0.6);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_det_db_unclip_ratio(ref MDModel model, double dbUnclipRatio = 1.5);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_det_db_set_use_dilation(ref MDModel model, int useDilation = 0);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_model_predict(ref MDModel model, ref MDImage image, ref MDOCRResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDRect md_get_text_position(ref MDModel model, ref MDImage image, string text);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_ocr_result(ref MDImage image, ref MDOCRResults results, string fontPath,
            int fontSize, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_result(ref MDOCRResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_model(ref MDModel model);
    }

    public class OcrRecognition : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public OcrRecognition(string modelPath, string dictPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            if (md_create_ocr_recognition_model(ref _model, modelPath, dictPath, ref nativeOption) != 0)
            {
                md_free_ocr_recognition_model(ref _model);
                throw new Exception("md_create_ocr_recognition_model failed.");
            }
        }

        public OcrResult Predict(Image image)
        {
            MDOCRResult native = new MDOCRResult();
            Utils.Check(md_ocr_recognition_model_predict(ref _model, ref image.RawImage, ref native),
                "md_ocr_recognition_model_predict");
            var result = OcrResult.FromNative(native);
            md_free_ocr_recognition_result(ref native);
            return result;
        }

        public OcrResults BatchPredict(Image image, List<Polygon> polygons, int batchSize)
        {
            MDOCRResults nativeResults = new MDOCRResults();
            MDPolygon[] nativePolygons = null;

            try
            {
                nativePolygons = Polygon.ToMDPolygonArray(polygons);
                Utils.Check(md_ocr_recognition_model_predict_batch(
                    ref _model,
                    ref image.RawImage,
                    batchSize,
                    ref nativePolygons[0],
                    polygons.Count,
                    ref nativeResults
                ), "md_ocr_recognition_model_predict_batch");

                return OcrResults.FromNativeArray(nativeResults);
            }
            finally
            {
                if (nativePolygons != null)
                {
                    foreach (var polygon in nativePolygons)
                        Polygon.FreeMDPolygon(polygon);
                }

                md_free_ocr_result(ref nativeResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_ocr_recognition_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~OcrRecognition() => Dispose();

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_ocr_recognition_model(ref MDModel model, string modelPath, string dictPath,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_recognition_model_predict(ref MDModel model, ref MDImage image,
            ref MDOCRResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_ocr_recognition_model_predict_batch(ref MDModel model, ref MDImage image,
            int batchSize, ref MDPolygon polygons, int size, ref MDOCRResults results);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_recognition_result(ref MDOCRResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_recognition_model(ref MDModel model);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_ocr_result(ref MDOCRResults results);
    }

    public class PaddleStructureTable : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public PaddleStructureTable(MDStructureTableModelParameters parameters, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeRuntimeOption = option.ToNative();
            if (md_create_structure_table_model(ref _model, ref parameters, ref nativeRuntimeOption) != 0)
                throw new Exception("md_create_ocr_model failed.");
        }

        public OcrResults Predict(Image image)
        {
            MDOCRResults cResults = new MDOCRResults();
            Utils.Check(md_structure_table_model_predict(ref _model, ref image.RawImage, ref cResults),
                "md_ocr_model_predict");
            var results = OcrResults.FromNativeArray(cResults);
            md_free_structure_table_result(ref cResults);
            return results;
        }

        public void DrawOcrResult(Image image, OcrResults results, string fontPath, Color color,
            int fontSize = 12, double alpha = 0.5, int saveResult = 1)
        {
            var nativeResults = OcrResults.ToNativeArray(results);
            var nativeColor = color.ToNative();
            md_draw_structure_table_result(ref image.RawImage, ref nativeResults, fontPath, fontSize, alpha,
                saveResult);
            md_free_structure_table_result(ref nativeResults);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_structure_table_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~PaddleStructureTable() => Dispose();

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_structure_table_model(ref MDModel model,
            ref MDStructureTableModelParameters parameters, ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_structure_table_model_predict(ref MDModel model, ref MDImage image,
            ref MDOCRResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_structure_table_result(ref MDImage image, ref MDOCRResults cResults,
            string fontPath, int fontSize, double alpha, int saveResult);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_structure_table_result(ref MDOCRResults cResults);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_structure_table_result(ref MDOCRResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_structure_table_model(ref MDModel model);
    }
}
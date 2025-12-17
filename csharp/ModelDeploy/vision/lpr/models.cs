using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.lpr
{
    public sealed class LprDetection : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public LprDetection(string modelDir, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_lpr_det_model(ref _model, modelDir, ref nativeOption), "Create detection model");
        }


        public List<face.KeyPointResult> Predict(Image image)
        {
            var cResults = new MDKeyPointResults();
            Utils.Check(md_lpr_det_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<face.KeyPointResult>(face.KeyPointResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_lpr_det_result(ref cResults);
            }
        }

        public void Display(List<face.KeyPointResult> results)
        {
            var cResults = face.KeyPointResult.ToNativeArray(results);
            try
            {
                md_print_lpr_det_result(ref cResults);
            }
            finally
            {
                md_free_lpr_det_result(ref cResults);
            }
        }

        public void DrawDetectionResult(Image image, List<face.KeyPointResult> results, string fontPath,
            int fontSize = 12, int landmarkRadius = 2, double alpha = 0.5, int saveResult = 1)
        {
            var cResults = face.KeyPointResult.ToNativeArray(results);
            try
            {
                md_draw_lpr_det_result(ref image.RawImage, ref cResults, fontPath, fontSize, landmarkRadius, alpha,
                    saveResult);
            }
            finally
            {
                md_free_lpr_det_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_lpr_det_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~LprDetection() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_lpr_det_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_lpr_det_predict(ref MDModel model, ref MDImage image,
            ref MDKeyPointResults cResults);


        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_lpr_det_result(ref MDImage image, ref MDKeyPointResults cResults,
            string fontPath, int fontSize, int landmarkRadius, double alpha, int saveResult);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_lpr_det_result(ref MDKeyPointResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_det_result(ref MDKeyPointResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_det_model(ref MDModel model);

        #endregion
    }

    public sealed class LprRecognizer : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public LprRecognizer(string modelDir, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_lpr_rec_model(ref _model, modelDir, ref nativeOption), "Create detection model");
        }


        public List<LprResult> Predict(Image image)
        {
            var cResults = new MDLPRResults();
            Utils.Check(md_lpr_rec_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<LprResult>(LprResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_lpr_rec_result(ref cResults);
            }
        }

        public void Display(List<LprResult> results)
        {
            var cResults = LprResult.ToNativeArray(results);
            try
            {
                md_print_lpr_rec_result(ref cResults);
            }
            finally
            {
                md_free_lpr_rec_result(ref cResults);
            }
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_lpr_rec_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~LprRecognizer() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_lpr_rec_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_lpr_rec_predict(ref MDModel model, ref MDImage image,
            ref MDLPRResults cResults);


        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_lpr_rec_result(ref MDLPRResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_rec_result(ref MDLPRResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_rec_model(ref MDModel model);

        #endregion
    }

    public sealed class LprPipeline : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public LprPipeline(string lprDetModelFile, string lprRecModelFile, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_lpr_pipeline_model(ref _model, lprDetModelFile, lprRecModelFile, ref nativeOption),
                "Create detection model");
        }


        public List<LprResult> Predict(Image image)
        {
            var cResults = new MDLPRResults();
            Utils.Check(md_lpr_pipeline_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<LprResult>(LprResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_lpr_pipeline_result(ref cResults);
            }
        }

        public void Display(List<LprResult> results)
        {
            var cResults = LprResult.ToNativeArray(results);
            try
            {
                md_print_lpr_pipeline_result(ref cResults);
            }
            finally
            {
                md_free_lpr_pipeline_result(ref cResults);
            }
        }

        public void DrawLprResult(Image image, List<LprResult> results, string fontPath,
            int fontSize = 12, int landmarkRadius = 2, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = LprResult.ToNativeArray(results);
            try
            {
                md_draw_lpr_pipeline_result(ref image.RawImage, ref cResults, fontPath, fontSize, landmarkRadius, alpha,
                    saveResult);
            }
            finally
            {
                md_free_lpr_pipeline_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_lpr_pipeline_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~LprPipeline() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_lpr_pipeline_model(ref MDModel model, string lprDetModelFile,
            string lprRecModelFile, ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_lpr_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDLPRResults cResults);


        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_lpr_pipeline_result(ref MDLPRResults cResults);


        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_lpr_pipeline_result(ref MDImage image, ref MDLPRResults cResults,
            string fontPath, int fontSize, int landmarkRadius, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_pipeline_result(ref MDLPRResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_lpr_pipeline_model(ref MDModel model);

        #endregion
    }
}
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.face
{
    public sealed class Scrfd : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public Scrfd(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_det_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public List<DetectionLandmarkResult> Predict(Image image)
        {
            var cResults = new MDDetectionLandmarkResults();
            Utils.Check(md_face_det_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<DetectionLandmarkResult>(DetectionLandmarkResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_face_det_result(ref cResults);
            }
        }

        public void DrawFaceResult(Image image, List<DetectionLandmarkResult> results, string fontPath,
            int fontSize = 12, int landmarkRadius = 2, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = DetectionLandmarkResult.ToNativeArray(results);
            try
            {
                md_draw_face_det_result(ref image.RawImage, ref cResults, fontPath, fontSize, landmarkRadius, alpha,
                    saveResult);
            }
            finally
            {
                md_free_face_det_result(ref cResults);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_det_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~Scrfd() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_face_det_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_det_predict(ref MDModel model, ref MDImage image,
            ref MDDetectionLandmarkResults cResults);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_face_det_result(ref MDImage image,
            ref MDDetectionLandmarkResults cResults, string fontPath, int fontSize,
            int landmarkRadius, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_det_result(ref MDDetectionLandmarkResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_det_model(ref MDModel model);

        #endregion
    }

    public sealed class SeetaFaceId : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public SeetaFaceId(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_rec_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public FaceRecognizerResult Predict(Image image)
        {
            var cResult = new MDFaceRecognizerResult();
            Utils.Check(md_face_rec_predict(ref _model, ref image.RawImage, ref cResult), "Detection predict");
            try
            {
                return FaceRecognizerResult.FromNative(cResult);
            }
            finally
            {
                md_free_face_rec_result(ref cResult);
            }
        }

        public void Display(FaceRecognizerResult result)
        {
            var cResult = FaceRecognizerResult.ToNative(result);
            try
            {
                md_print_face_rec_result(ref cResult);
            }
            finally
            {
                md_free_face_rec_result(ref cResult);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_rec_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceId() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_face_rec_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_rec_predict(ref MDModel model, ref MDImage image,
            ref MDFaceRecognizerResult cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_face_rec_result(ref MDFaceRecognizerResult cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_rec_result(ref MDFaceRecognizerResult cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_rec_model(ref MDModel model);

        #endregion
    }


    public sealed class SeetaFaceAge : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public SeetaFaceAge(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_age_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public int Predict(Image image)
        {
            int age = 0;
            Utils.Check(md_face_age_predict(ref _model, ref image.RawImage, ref age), "Detection predict");
            return age;
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_age_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceAge() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_face_age_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_age_predict(ref MDModel model, ref MDImage image,
            ref int cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_age_model(ref MDModel model);

        #endregion
    }

    public sealed class SeetaFaceGender : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public enum Gender
        {
            Female,
            Male
        }

        public SeetaFaceGender(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_gender_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public Gender Predict(Image image)
        {
            int gender = 0;
            Utils.Check(md_face_gender_predict(ref _model, ref image.RawImage, ref gender), "Detection predict");
            return (Gender)gender;
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_gender_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceGender() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_face_gender_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_gender_predict(ref MDModel model, ref MDImage image,
            ref int cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_gender_model(ref MDModel model);

        #endregion
    }

    public sealed class SeetaFaceAsFirst : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public SeetaFaceAsFirst(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_as_first_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public float Predict(Image image)
        {
            float score = 0;
            Utils.Check(md_face_as_first_predict(ref _model, ref image.RawImage, ref score), "Detection predict");
            return score;
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_as_first_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceAsFirst() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_face_as_first_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_as_first_predict(ref MDModel model, ref MDImage image, ref float cResult);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_as_first_model(ref MDModel model);

        #endregion
    }

    public sealed class SeetaFaceAsSecond : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public SeetaFaceAsSecond(string modelDir, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_as_second_model(ref _model, modelDir, threadNum), "Create detection model");
        }


        public List<FaceAsSecondResult> Predict(Image image)
        {
            var cResults = new MDFaceAsSecondResults();
            Utils.Check(md_face_as_second_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<FaceAsSecondResult>(FaceAsSecondResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_face_as_second_result(ref cResults);
            }
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_as_second_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceAsSecond() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int
            md_create_face_as_second_model(ref MDModel model, string modelPath, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_as_second_predict(ref MDModel model, ref MDImage image,
            ref MDFaceAsSecondResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_as_second_result(ref MDFaceAsSecondResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_as_second_model(ref MDModel model);

        #endregion
    }

    public sealed class SeetaFaceAntiSpoof : IDisposable
    {
        private MDModel _model;
        private bool _disposed;


        public SeetaFaceAntiSpoof(string faceDetModelFile, string firstModelFile, string secondModelFile,
            int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_as_pipeline_model(ref _model, faceDetModelFile, firstModelFile, secondModelFile,
                threadNum), "Create detection model");
        }


        public List<MDFaceAsResult> Predict(Image image)
        {
            var cResults = new MDFaceAsResults();
            Utils.Check(md_face_as_pipeline_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                var results = new List<MDFaceAsResult>();
                for (var i = 0; i < cResults.size; i++)
                {
                    var currentPtr = IntPtr.Add(cResults.data, i * sizeof(int));
                    results.Add((MDFaceAsResult)Marshal.ReadInt32(currentPtr));
                }

                return results;
            }
            finally
            {
                md_free_face_as_pipeline_result(ref cResults);
            }
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_as_pipeline_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~SeetaFaceAntiSpoof() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int
            md_create_face_as_pipeline_model(ref MDModel model, string faceDetModelFile, string firstModelFile,
                string secondModelFile, int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_as_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDFaceAsResults cResults, float fuseThreshold = 0.8f, float clarityThreshold = 0.3f);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_as_pipeline_result(ref MDFaceAsResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_as_pipeline_model(ref MDModel model);

        #endregion
    }

    public sealed class FaceRecognizerPipeline : IDisposable
    {
        private MDModel _model;
        private bool _disposed;


        public FaceRecognizerPipeline(string faceDetModelFile, string faceRecModelFile, int threadNum = 8)
        {
            _model = new MDModel();
            Utils.Check(md_create_face_rec_pipeline_model(ref _model, faceDetModelFile, faceRecModelFile, threadNum),
                "Create detection model");
        }


        public List<FaceRecognizerResult> Predict(Image image)
        {
            var cResults = new MDFaceRecognizerResults();
            Utils.Check(md_face_rec_pipeline_predict(ref _model, ref image.RawImage, ref cResults),
                "Detection predict");
            try
            {
                return new List<FaceRecognizerResult>(FaceRecognizerResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_face_rec_pipeline_result(ref cResults);
            }
        }

        public void Display(List<FaceRecognizerResult> results)
        {
            var cResult = FaceRecognizerResult.ToNativeArray(results);
            md_print_face_rec_pipeline_result(ref cResult);
        }


        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_face_rec_pipeline_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~FaceRecognizerPipeline() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int
            md_create_face_rec_pipeline_model(ref MDModel model, string faceDetModelFile, string faceRecModelFile,
                int threadNum = 8);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_face_rec_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDFaceRecognizerResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_face_rec_pipeline_result(ref MDFaceRecognizerResults cResults);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_rec_pipeline_result(ref MDFaceRecognizerResults cResults);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_face_rec_pipeline_model(ref MDModel model);

        #endregion
    }
}
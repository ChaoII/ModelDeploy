using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;
using static ModelDeploy.NativeMethods;
using ModelDeploy.vision.face;

namespace ModelDeploy.vision.pose
{
    public sealed class UltralyticsPose : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        internal UltralyticsPose(MDModel existing)
        {
            _model = existing;
        }

        public UltralyticsPose(string modelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_keypoint_model(ref _model, modelPath, ref nativeOption), "Create detection model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_keypoint_input_size(ref _model, size), "Set detection input size");
        }

        public List<KeyPointResult> Predict(Image image)
        {
            var cResults = new MDKeyPointResults();
            Utils.Check(md_keypoint_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<KeyPointResult>(KeyPointResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }


        public List<KeyPointResult> PredictNv12(byte[] srcY, byte[] srcUV, int width, int height,
            int stepY, int stepUV, Device srcDevice = Device.CPU)
        {
            var cResults = new MDKeyPointResults();
            var yPinned = GCHandle.Alloc(srcY, GCHandleType.Pinned);
            var uvPinned = GCHandle.Alloc(srcUV, GCHandleType.Pinned);
            try
            {
                Utils.Check(md_pose_predict_nv12(ref _model, yPinned.AddrOfPinnedObject(),
                        uvPinned.AddrOfPinnedObject(), width, height, stepY, stepUV, srcDevice, ref cResults),
                    "Pose predict NV12");
                return new List<KeyPointResult>(KeyPointResult.FromNativeArray(cResults));
            }
            finally
            {
                yPinned.Free();
                uvPinned.Free();
                md_free_keypoint_result(ref cResults);
            }
        }

        public void Display(List<KeyPointResult> results)
        {
            var cResults = KeyPointResult.ToNativeArray(results);
            try
            {
                md_print_keypoint_result(ref cResults);
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }


        public void DrawPoseResult(Image image, List<KeyPointResult> results, string fontPath, int fontSize = 12,
            int keypointRadius = 4, double alpha = 0.5, bool saveResult = false)
        {
            var cResults = KeyPointResult.ToNativeArray(results);
            try
            {
                md_draw_keypoint_result(ref image.RawImage, ref cResults, fontPath, fontSize,
                    keypointRadius, alpha, saveResult);
            }
            finally
            {
                md_free_keypoint_result(ref cResults);
            }
        }

        public UltralyticsPose Clone()
        {
            var clone = new MDModel();
            Utils.Check(md_clone_model(ref clone, ref _model), "Clone keypoint model");
            return new UltralyticsPose(clone);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_keypoint_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsPose() => Dispose();

    }
}

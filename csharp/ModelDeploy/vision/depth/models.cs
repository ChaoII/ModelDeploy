using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.vision.depth
{
    public sealed class UltralyticsDepth : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        internal UltralyticsDepth(MDModel existing)
        {
            _model = existing;
        }

        public UltralyticsDepth(string modelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_depth_model(ref _model, modelPath, ref nativeOption),
                "Create depth estimation model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_depth_input_size(ref _model, size), "Set depth estimation input size");
        }

        public DepthResult Predict(Image image)
        {
            var cResult = new MDDepthResult();
            Utils.Check(md_depth_predict(ref _model, ref image.RawImage, ref cResult),
                "Depth estimation predict");
            try
            {
                return DepthResult.FromNative(cResult);
            }
            finally
            {
                md_free_depth_result(ref cResult);
            }
        }

        public DepthResult PredictNv12(byte[] srcY, byte[] srcUV, int width, int height,
            int stepY, int stepUV, MDDevice srcDevice = MDDevice.CPU)
        {
            var cResult = new MDDepthResult();
            var yPinned = GCHandle.Alloc(srcY, GCHandleType.Pinned);
            var uvPinned = GCHandle.Alloc(srcUV, GCHandleType.Pinned);
            try
            {
                Utils.Check(md_depth_predict_nv12(ref _model, yPinned.AddrOfPinnedObject(),
                        uvPinned.AddrOfPinnedObject(), width, height, stepY, stepUV, srcDevice, ref cResult),
                    "Depth estimation predict NV12");
                return DepthResult.FromNative(cResult);
            }
            finally
            {
                yPinned.Free();
                uvPinned.Free();
                md_free_depth_result(ref cResult);
            }
        }

        public UltralyticsDepth Clone()
        {
            var clone = new MDModel();
            Utils.Check(md_clone_model(ref clone, ref _model), "Clone depth estimation model");
            return new UltralyticsDepth(clone);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_depth_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsDepth() => Dispose();
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.vision.sem
{
    public sealed class UltralyticsSem : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        internal UltralyticsSem(MDModel existing)
        {
            _model = existing;
        }

        public UltralyticsSem(string modelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_sem_model(ref _model, modelPath, ref nativeOption),
                "Create semantic segmentation model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_sem_input_size(ref _model, size), "Set semantic segmentation input size");
        }

        public SemSegResult Predict(Image image)
        {
            var cResult = new MDSemSegResult();
            Utils.Check(md_sem_predict(ref _model, ref image.RawImage, ref cResult),
                "Semantic segmentation predict");
            try
            {
                return SemSegResult.FromNative(cResult);
            }
            finally
            {
                md_free_sem_result(ref cResult);
            }
        }

        public SemSegResult PredictNv12(byte[] srcY, byte[] srcUV, int width, int height,
            int stepY, int stepUV, Device srcDevice = Device.CPU)
        {
            var cResult = new MDSemSegResult();
            var yPinned = GCHandle.Alloc(srcY, GCHandleType.Pinned);
            var uvPinned = GCHandle.Alloc(srcUV, GCHandleType.Pinned);
            try
            {
                Utils.Check(md_sem_predict_nv12(ref _model, yPinned.AddrOfPinnedObject(),
                        uvPinned.AddrOfPinnedObject(), width, height, stepY, stepUV, srcDevice, ref cResult),
                    "Semantic segmentation predict NV12");
                return SemSegResult.FromNative(cResult);
            }
            finally
            {
                yPinned.Free();
                uvPinned.Free();
                md_free_sem_result(ref cResult);
            }
        }

        public UltralyticsSem Clone()
        {
            var clone = new MDModel();
            Utils.Check(md_clone_model(ref clone, ref _model), "Clone semantic segmentation model");
            return new UltralyticsSem(clone);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_sem_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsSem() => Dispose();
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

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
            Utils.Check(NativeBindings.md_create_depth_model(ref _model, modelPath, ref nativeOption),
                "Create depth estimation model");
        }

        public void SetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(NativeBindings.md_set_depth_input_size(ref _model, size), "Set depth estimation input size");
        }

        public DepthResult Predict(Image image)
        {
            var cResult = new MDDepthResult();
            Utils.Check(NativeBindings.md_depth_predict(ref _model, ref image.RawImage, ref cResult),
                "Depth estimation predict");
            try
            {
                return DepthResult.FromNative(cResult);
            }
            finally
            {
                NativeBindings.md_free_depth_result(ref cResult);
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
                NativeBindings.md_free_depth_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~UltralyticsDepth() => Dispose();

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_clone_model(ref MDModel model, ref MDModel from);
    }
}

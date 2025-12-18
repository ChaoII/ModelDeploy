using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.pipeline
{
    public sealed class PedestrianAttribute : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        public PedestrianAttribute(string detModelPath, string clsModelPath, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeRuntimeOption = option.ToNative();
            Utils.Check(md_create_attr_model(ref _model, detModelPath, clsModelPath, ref nativeRuntimeOption),
                "Create PedestrianAttribute model");
        }

        public void SetDetInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_attr_det_input_size(ref _model, size), "Set detection input size");
        }

        public void SetClsInputSize(int width, int height)
        {
            var size = new MDSize { width = width, height = height };
            Utils.Check(md_set_attr_cls_input_size(ref _model, size), "Set cls input size");
        }

        public void SetClsBatchSize(int batchSize)
        {
            Utils.Check(md_set_attr_cls_batch_size(ref _model, batchSize), "Set cls batch size");
        }


        public void SetDetThreshold(float threshold)
        {
            Utils.Check(md_set_attr_det_threshold(ref _model, threshold), "Set det threshold");
        }


        public List<AttributeResult> Predict(Image image)
        {
            var cResults = new MDAttributeResults();
            Utils.Check(md_attr_model_predict(ref _model, ref image.RawImage, ref cResults), "Detection predict");
            try
            {
                return new List<AttributeResult>(AttributeResult.FromNativeArray(cResults));
            }
            finally
            {
                md_free_attr_result(ref cResults);
            }
        }


        public void Display(List<AttributeResult> results)
        {
            var cResults = AttributeResult.ToNativeArray(results);
            try
            {
                md_print_attr_result(ref cResults);
            }
            finally
            {
                md_free_attr_result(ref cResults);
            }
        }


        public void DrawAttributeResult(Image image, List<AttributeResult> results,
            double threshold, Dictionary<int, string> labelMap, string fontPath, int fontSize = 12,
            double alpha = 0.5, bool saveResult = false)
        {
            var cResults = AttributeResult.ToNativeArray(results);
            var cMap = Utils.DictionaryToMDMapData(labelMap);

            try
            {
                md_draw_attr_result(ref image.RawImage, ref cResults,
                    threshold, ref cMap, fontPath, fontSize, alpha, saveResult);
            }
            finally
            {
                md_free_attr_result(ref cResults);
                Utils.md_free_md_map(ref cMap);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_attr_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~PedestrianAttribute() => Dispose();

        #region Native bindings

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_attr_model(
            ref MDModel model, string detModelPath, string clsModelPath, ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_attr_det_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_attr_cls_input_size(ref MDModel model, MDSize size);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_attr_cls_batch_size(ref MDModel model, int batchSize);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_set_attr_det_threshold(ref MDModel model, float threshold);


        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_attr_model_predict(ref MDModel model, ref MDImage image,
            ref MDAttributeResults results);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_print_attr_result(ref MDAttributeResults cResults);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_attr_result(ref MDImage image, ref MDAttributeResults result,
            double threshold, ref MDMapData labelMap, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_attr_result(ref MDAttributeResults results);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_attr_model(ref MDModel model);

        #endregion
    }
}
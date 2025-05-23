using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.ocr
{
    public class OcrResult
    {
        private Polygon Box { get; set; }
        public string Text { get; private set; }
        private float Score { get; set; }

        public override string ToString() => $"Box: {Box}, Text: {Text}, Score: {Score}";

        public static OcrResult FromNative(MDOCRResult cResult)
        {
            return new OcrResult
            {
                Box = Polygon.FromNative(cResult.box),
                Text = Utils.PtrToStringUTF8(cResult.text) ?? string.Empty,
                Score = cResult.score
            };
        }

        private static MDOCRResult ToNative(OcrResult result)
        {
            return new MDOCRResult
            {
                box = result.Box.CopyToMDPolygon(), // native内存交由C释放
                text = Utils.ConvertStringToHGlobalUtf8(result.Text), // native内存交由C释放
                score = result.Score
            };
        }

        public static List<OcrResult> FromNativeArray(MDOCRResults cResults)
        {
            int size = cResults.size;
            var results = new List<OcrResult>(size);
            int structSize = Marshal.SizeOf<MDOCRResult>();

            for (int i = 0; i < size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResults.data, i * structSize);
                var nativeResult = Marshal.PtrToStructure<MDOCRResult>(currentPtr);
                results.Add(FromNative(nativeResult));
            }

            return results;
        }

        public static MDOCRResults ToNativeArray(List<OcrResult> results)
        {
            int count = results.Count;
            int structSize = Marshal.SizeOf<MDOCRResult>();
            IntPtr nativeArray = Marshal.AllocHGlobal(count * structSize);

            for (int i = 0; i < count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(nativeArray, i * structSize);
                MDOCRResult nativeResult = ToNative(results[i]);
                Marshal.StructureToPtr(nativeResult, currentPtr, false);
            }

            return new MDOCRResults
            {
                size = count,
                data = nativeArray
            };
        }
    }
}
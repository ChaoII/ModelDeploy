using System;
using System.Collections.Generic;
using System.Linq;
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
        private Polygon TableBox { get; set; }
        public string TableStructure { get; private set; }


        public override string ToString() => $"Box: {Box}, Text: {Text}, Score: {Score}";

        public static OcrResult FromNative(MDOCRResult cResult)
        {
            return new OcrResult
            {
                Box = Polygon.FromNative(cResult.box),
                Text = Utils.PtrToStringUTF8(cResult.text) ?? string.Empty,
                Score = cResult.score,
                TableBox = Polygon.FromNative(cResult.table_boxes),
                TableStructure = Utils.PtrToStringUTF8(cResult.table_structure) ?? string.Empty,
            };
        }

        public static MDOCRResult ToNative(OcrResult result)
        {
            return new MDOCRResult
            {
                box = result.Box.CopyToMDPolygon(), // native内存交由C释放
                text = Utils.ConvertStringToHGlobalUtf8(result.Text), // native内存交由C释放
                score = result.Score,
                table_boxes = result.TableBox.CopyToMDPolygon(),
                table_structure = Utils.ConvertStringToHGlobalUtf8(result.TableStructure)
            };
        }
    }

    public class OcrResults
    {
        public List<OcrResult> Data { get; private set; }
        public string TableHtml { get; private set; }

        public static OcrResults FromNativeArray(MDOCRResults cResults)
        {
            int size = cResults.size;
            var results = new OcrResults();
            results.Data = new List<OcrResult>(size);
            int structSize = Marshal.SizeOf<MDOCRResult>();

            for (int i = 0; i < size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResults.data, i * structSize);
                var nativeResult = Marshal.PtrToStructure<MDOCRResult>(currentPtr);
                results.Data.Add(OcrResult.FromNative(nativeResult));
            }

            results.TableHtml = Utils.PtrToStringUTF8(cResults.table_html) ?? string.Empty;
            return results;
        }

        public static MDOCRResults ToNativeArray(OcrResults results)
        {
            int count = results.Data.Count;
            int structSize = Marshal.SizeOf<MDOCRResult>();
            IntPtr nativeArray = Marshal.AllocHGlobal(count * structSize);

            for (int i = 0; i < count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(nativeArray, i * structSize);
                MDOCRResult nativeResult = OcrResult.ToNative(results.Data[i]);
                Marshal.StructureToPtr(nativeResult, currentPtr, false);
            }

            return new MDOCRResults
            {
                size = count,
                data = nativeArray,
                table_html = Utils.ConvertStringToHGlobalUtf8(results.TableHtml)
            };
        }
    }
}
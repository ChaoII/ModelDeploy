using System.Runtime.InteropServices;
using ModelDeploy.utils;
using ModelDeploy.types_internal_c;
using System.Collections.Generic;
using System;

namespace ModelDeploy.vision.ocr
{
    public class OCRResult
    {
        public Polygon Box { get; set; }
        public string Text { get; set; }
        public float Score { get; set; }


        public override string ToString()
        {
            return $"Box: {Box}, Text: {Text}, Score: {Score}";
        }

        public static OCRResult FromMDOCRResult(MDOCRResult cResult)
        {
            OCRResult result = new OCRResult
            {
                Box = Polygon.FromMDPolygon(cResult.box),
                Text = Utils.PtrToStringUTF8(cResult.text) ?? "",
                Score = cResult.score
            };
            return result;
        }

        private static MDOCRResult ToMDOCRResult(OCRResult result)
        {
            MDOCRResult cResult = new MDOCRResult
            {
                // notice
                // 此处开辟了内存，并将内存地址赋值给cresult.box, 因此需要在释放时进行释放(但是该处释放内存在C中进行)
                box = result.Box.CopyToMDPolygon(),
                // 在C中进行释放 无需Marshal.FreeHGlobal(cresult.text);
                text = Utils.ConvertStringToHGlobalUtf8(result.Text),
                score = result.Score
            };
            return cResult;
        }

        public static List<OCRResult> FromMDOCRResults(MDOCRResults cresults)
        {
            List<OCRResult> results = new List<OCRResult>();
            for (int i = 0; i < cresults.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cresults.data, i * Marshal.SizeOf<MDOCRResult>());
                MDOCRResult res = Marshal.PtrToStructure<MDOCRResult>(currentPtr);
                results.Add(FromMDOCRResult(res));
            }

            return results;
        }

        public static MDOCRResults ToMDOCRResults(List<OCRResult> results)
        {
            MDOCRResults cresults = new MDOCRResults
            {
                size = results.Count,
                // 在C中进行释放
                data = Marshal.AllocHGlobal(results.Count * Marshal.SizeOf<MDOCRResult>())
            };
            for (int i = 0; i < results.Count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cresults.data, i * Marshal.SizeOf<MDOCRResult>());
                MDOCRResult res = ToMDOCRResult(results[i]);
                Marshal.StructureToPtr(res, currentPtr, false);
            }

            return cresults;
        }
    }
}
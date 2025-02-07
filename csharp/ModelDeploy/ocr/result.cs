using System.Runtime.InteropServices;
using ModelDeploy.utils;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.ocr;

public class OCRResult
{
    public required Polygon Box { get; set; }
    public required string Text { get; set; }
    public float Score { get; set; }
    
    
    private static OCRResult FromMDOCRResult(MDOCRResult cresult)
    {
        OCRResult result = new OCRResult
        {
            Box = Polygon.FromRaw(cresult.box),
            Text = Marshal.PtrToStringUTF8(cresult.text) ?? "",
            Score = cresult.score
        };
        return result;
    }

    private static MDOCRResult ToMDOCRResult(OCRResult result)
    {
        MDOCRResult cresult = new MDOCRResult
        {
            box = result.Box.ToRaw(),
            // 在C中进行释放 无需Marshal.FreeHGlobal(cresult.text);
            text = Utils.ConvertStringToHGlobalUtf8(result.Text),
            score = result.Score
        };
        return cresult;
    }

    public static List<OCRResult> FromMDOCRResults(MDOCRResults cresults)
    {
        List<OCRResult> results = [];
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
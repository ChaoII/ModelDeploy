using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.detection
{
    public class DetectionResult
    {
        public Rect Box { get; set; }

        public int LabelId { get; set; }

        public float Score { get; set; }


        private static DetectionResult FromMDDetectionResult(MDDetectionResult cresult)
        {
            DetectionResult result = new DetectionResult
            {
                Box = Rect.FromRaw(cresult.box),
                LabelId = cresult.label_id,
                Score = cresult.score
            };
            return result;
        }

        private static MDDetectionResult ToMDDetectionResult(DetectionResult result)
        {
            MDDetectionResult cresult = new MDDetectionResult
            {
                box = result.Box.ToRaw(),
                label_id = result.LabelId,
                score = result.Score
            };
            return cresult;
        }

        public static List<DetectionResult> FromMDDetectionResults(MDDetectionResults cresults)
        {
            List<DetectionResult> results = new List<DetectionResult>();
            for (int i = 0; i < cresults.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cresults.data, i * Marshal.SizeOf<MDDetectionResult>());
                MDDetectionResult res = Marshal.PtrToStructure<MDDetectionResult>(currentPtr);
                results.Add(FromMDDetectionResult(res));
            }

            return results;
        }

        public static MDDetectionResults ToMDDetectionResults(List<DetectionResult> results)
        {
            MDDetectionResults cresults = new MDDetectionResults
            {
                size = results.Count
            };
            if (results.Count > 0)
            {
                cresults.data = Marshal.AllocHGlobal(results.Count * Marshal.SizeOf<MDDetectionResult>());
                for (int i = 0; i < results.Count; i++)
                {
                    IntPtr currentPtr = IntPtr.Add(cresults.data, i * Marshal.SizeOf<MDDetectionResult>());
                    MDDetectionResult res = ToMDDetectionResult(results[i]);
                    Marshal.StructureToPtr(res, currentPtr, false);
                }
            }

            return cresults;
        }

        public static void FreeMDDetectionResults(MDDetectionResults cresults)
        {
            if (cresults.data != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(cresults.data);
            }
        }
    }
}
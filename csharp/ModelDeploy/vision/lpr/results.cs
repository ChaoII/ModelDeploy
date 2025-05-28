using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.vision.lpr
{
    public class LprResult
    {
        public Rect Box { get; set; }
        public List<Point> Landmarks { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }

        public string CarPlateStr { get; private set; }

        public string CarPlateColor { get; private set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDLPRResult>();

        private static LprResult FromNative(MDLPRResult cResult)
        {
            var lprResult = new LprResult
            {
                Box = Rect.FromNative(cResult.box),
                LabelId = cResult.label_id,
                Score = cResult.score,
                Landmarks = new List<Point>(cResult.landmarks_size),
                CarPlateStr = Utils.PtrToStringUTF8(cResult.car_plate_str),
                CarPlateColor = Utils.PtrToStringUTF8(cResult.car_plate_color)
            };
            for (var i = 0; i < cResult.landmarks_size; i++)
            {
                var currentPtr = IntPtr.Add(cResult.landmarks, i * Marshal.SizeOf<MDPoint>());
                var cPoint = Marshal.PtrToStructure<MDPoint>(currentPtr);
                lprResult.Landmarks.Add(Point.FromNative(cPoint));
            }

            return lprResult;
        }

        private static MDLPRResult ToNative(LprResult result)
        {
            var cResult = new MDLPRResult
            {
                box = result.Box.ToNative(),
                label_id = result.LabelId,
                score = result.Score,
                landmarks_size = result.Landmarks.Count,
                landmarks = result.Landmarks.Count > 0
                    ? Marshal.AllocHGlobal(result.Landmarks.Count * Marshal.SizeOf<MDPoint>())
                    : IntPtr.Zero,
                car_plate_str = Utils.ConvertStringToHGlobalUtf8(result.CarPlateStr),
                car_plate_color = Utils.ConvertStringToHGlobalUtf8(result.CarPlateColor)
            };
            for (var i = 0; i < result.Landmarks.Count; i++)
            {
                var currentPtr = IntPtr.Add(cResult.landmarks, i * Marshal.SizeOf<MDPoint>());
                Marshal.StructureToPtr(result.Landmarks[i].ToNative(), currentPtr, false);
            }

            return cResult;
        }

        public static List<LprResult> FromNativeArray(MDLPRResults cResults)
        {
            var results = new List<LprResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDLPRResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDLPRResults ToNativeArray(List<LprResult> results)
        {
            var cResults = new MDLPRResults
            {
                size = results.Count,
                data = results.Count > 0
                    ? Marshal.AllocHGlobal(results.Count * NativeSize)
                    : IntPtr.Zero
            };

            for (int i = 0; i < results.Count; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = ToNative(results[i]);
                Marshal.StructureToPtr(native, ptr, false);
            }

            return cResults;
        }
    }
}
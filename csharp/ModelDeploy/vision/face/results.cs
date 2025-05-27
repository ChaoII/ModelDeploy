using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.face
{
    public class DetectionLandmarkResult
    {
        public Rect Box { get; set; }
        public List<Point> Landmarks { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDDetectionLandmarkResult>();

        private static DetectionLandmarkResult FromNative(MDDetectionLandmarkResult cResult)
        {
            var detResult = new DetectionLandmarkResult
            {
                Box = Rect.FromNative(cResult.box),
                LabelId = cResult.label_id,
                Score = cResult.score,
                Landmarks = new List<Point>(cResult.landmarks_size)
            };
            for (var i = 0; i < cResult.landmarks_size; i++)
            {
                var currentPtr = IntPtr.Add(cResult.landmarks_data, i * Marshal.SizeOf<MDPoint>());
                var cPoint = Marshal.PtrToStructure<MDPoint>(currentPtr);
                detResult.Landmarks.Add(Point.FromNative(cPoint));
            }

            return detResult;
        }

        private static MDDetectionLandmarkResult ToNative(DetectionLandmarkResult result)
        {
            var cResult = new MDDetectionLandmarkResult
            {
                box = result.Box.ToNative(),
                label_id = result.LabelId,
                score = result.Score,
                landmarks_size = result.Landmarks.Count,
                landmarks_data = result.Landmarks.Count > 0
                    ? Marshal.AllocHGlobal(result.Landmarks.Count * Marshal.SizeOf<MDPoint>())
                    : IntPtr.Zero
            };
            for (var i = 0; i < result.Landmarks.Count; i++)
            {
                var currentPtr = IntPtr.Add(cResult.landmarks_data, i * Marshal.SizeOf<MDPoint>());
                Marshal.StructureToPtr(result.Landmarks[i].ToNative(), currentPtr, false);
            }

            return cResult;
        }

        public static List<DetectionLandmarkResult> FromNativeArray(MDDetectionLandmarkResults cResults)
        {
            var results = new List<DetectionLandmarkResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDDetectionLandmarkResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDDetectionLandmarkResults ToNativeArray(List<DetectionLandmarkResult> results)
        {
            var cResults = new MDDetectionLandmarkResults
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

    public class FaceRecognizerResult
    {
        public List<float> Embedding { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDFaceRecognizerResult>();

        public static FaceRecognizerResult FromNative(MDFaceRecognizerResult cResult)
        {
            var detResult = new FaceRecognizerResult
            {
                Embedding = new List<float>(cResult.size)
            };

            float[] buffer = new float[cResult.size];
            Marshal.Copy(cResult.embedding, buffer, 0, cResult.size);
            detResult.Embedding = new List<float>(buffer);
            return detResult;
        }

        public static MDFaceRecognizerResult ToNative(FaceRecognizerResult result)
        {
            var cResult = new MDFaceRecognizerResult
            {
                size = result.Embedding.Count,
                embedding = Marshal.AllocHGlobal(result.Embedding.Count * sizeof(float))
            };

            float[] buffer = result.Embedding.ToArray();
            Marshal.Copy(buffer, 0, cResult.embedding, buffer.Length);
            return cResult;
        }


        public static List<FaceRecognizerResult> FromNativeArray(MDFaceRecognizerResults cResults)
        {
            var results = new List<FaceRecognizerResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDFaceRecognizerResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDFaceRecognizerResults ToNativeArray(List<FaceRecognizerResult> results)
        {
            var cResults = new MDFaceRecognizerResults
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

    public class FaceAsSecondResult
    {
        private int LabelId { get; set; }
        private float Score { get; set; }

        public override string ToString() => $"LabelId: {LabelId}, Score: {Score}";

        private static readonly int NativeSize = Marshal.SizeOf<MDFaceAsSecondResult>();

        private static FaceAsSecondResult FromNative(MDFaceAsSecondResult cResult) => new FaceAsSecondResult
        {
            LabelId = cResult.label_id,
            Score = cResult.score
        };

        private static MDFaceAsSecondResult ToNative(FaceAsSecondResult result) => new MDFaceAsSecondResult
        {
            label_id = result.LabelId,
            score = result.Score
        };

        public static List<FaceAsSecondResult> FromNativeArray(MDFaceAsSecondResults cResults)
        {
            var results = new List<FaceAsSecondResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDFaceAsSecondResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDFaceAsSecondResults ToNativeArray(List<FaceAsSecondResult> results)
        {
            var cResults = new MDFaceAsSecondResults
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
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.vision.face
{
    public class KeyPointResult
    {
        public Rect Box { get; set; }
        public List<Point3F> Landmarks { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }

        private static readonly int NativeSize = Marshal.SizeOf<MDKeyPointResult>();

        private static KeyPointResult FromNative(MDKeyPointResult cResult)
        {
            var detResult = new KeyPointResult
            {
                Box = Rect.FromNative(cResult.box),
                LabelId = cResult.label_id,
                Score = cResult.score,
                Landmarks = new List<Point3F>(cResult.keypoints_size)
            };
            for (var i = 0; i < cResult.keypoints_size; i++)
            {
                var currentPtr = IntPtr.Add(cResult.keypoints_data, i * Marshal.SizeOf<MDPoint3f>());
                var cPoint = Marshal.PtrToStructure<MDPoint3f>(currentPtr);
                detResult.Landmarks.Add(Point3F.FromNative(cPoint));
            }

            return detResult;
        }

        private static MDKeyPointResult ToNative(KeyPointResult result)
        {
            var cResult = new MDKeyPointResult
            {
                box = result.Box.ToNative(),
                label_id = result.LabelId,
                score = result.Score,
                keypoints_size = result.Landmarks.Count,
                keypoints_data = result.Landmarks.Count > 0
                    ? Marshal.AllocHGlobal(result.Landmarks.Count * Marshal.SizeOf<MDPoint3f>())
                    : IntPtr.Zero
            };
            for (var i = 0; i < result.Landmarks.Count; i++)
            {
                var currentPtr = IntPtr.Add(cResult.keypoints_data, i * Marshal.SizeOf<MDPoint3f>());
                Marshal.StructureToPtr(result.Landmarks[i].ToNative(), currentPtr, false);
            }

            return cResult;
        }

        public static List<KeyPointResult> FromNativeArray(MDKeyPointResults cResults)
        {
            var results = new List<KeyPointResult>(cResults.size);
            for (int i = 0; i < cResults.size; i++)
            {
                IntPtr ptr = IntPtr.Add(cResults.data, i * NativeSize);
                var native = Marshal.PtrToStructure<MDKeyPointResult>(ptr);
                results.Add(FromNative(native));
            }

            return results;
        }

        public static MDKeyPointResults ToNativeArray(List<KeyPointResult> results)
        {
            var cResults = new MDKeyPointResults
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
            var detResult = new FaceRecognizerResult();
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
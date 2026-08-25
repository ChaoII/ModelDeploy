using System;
using System.Linq;
using System.Runtime.InteropServices;
using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;
using NUnit.Framework;

namespace ModelDeployUnitTest;

/// <summary>
/// NV12 设备直通（GPU 零拷贝）与 host 路径的一致性测试。
/// 针对 PedestrianAttribute 管线（det + 属性分类），验证：
///   - host NV12 与 device NV12（GPU 零拷贝）产出相同的检测数与属性；
///   - 先跑 host 预测再跑 device 预测时 device 不被竞速破坏（回归原跨流同步 Bug）；
///   - device 重复调用结果稳定。
/// 依赖 TRT(GPU) + .engine 测试模型；缺失或无 GPU 时 Ignore。
/// </summary>
public class Nv12DevicePassthroughTests
{
    private static string? _engineDir;
    private static bool _engineResolved;

    private static string EngineDir
    {
        get
        {
            if (_engineResolved) return _engineDir!;
            _engineResolved = true;
            var env = Environment.GetEnvironmentVariable("PEDATTR_ENGINE_DIR");
            if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
            {
                _engineDir = env;
                return env;
            }
            var testData = Environment.GetEnvironmentVariable("TEST_DATA_DIR")
                           ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
            var fallback = Path.Combine(testData, "test_data", "test_models", "trt", "pedattr");
            if (Directory.Exists(fallback)) _engineDir = fallback;
            return _engineDir!;
        }
    }

    private static string DetEngine => Path.Combine(EngineDir, "zhgd_det_20260629.engine");
    private static string ClsEngine => Path.Combine(EngineDir, "zhgd_ml_20260629.engine");

    private static string TestDataRoot =>
        Environment.GetEnvironmentVariable("TEST_DATA_DIR")
        ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");

    private static string TestImage => Path.Combine(TestDataRoot, "test_data", "test_images",
        "test_pedestrian_attribute_scale.jpg");

    private static bool HasGpuRuntime()
    {
        try { return Nv12Cudart.cudaDeviceCount(out var n) == 0 && n > 0; }
        catch { return false; }
    }

    [SetUp]
    public void SetUp()
    {
        if (string.IsNullOrEmpty(EngineDir) || !System.IO.File.Exists(DetEngine) || !System.IO.File.Exists(ClsEngine))
            Assert.Ignore($"pedattr TRT engines not found ({EngineDir}); set PEDATTR_ENGINE_DIR");
        if (!System.IO.File.Exists(TestImage))
            Assert.Ignore($"test image not found: {TestImage}");
        if (!HasGpuRuntime())
            Assert.Ignore("no CUDA GPU runtime available");
    }

    // BGR 像素 -> NV12（BT.601 limited，UV 偶数采样）
    private static (byte[] y, byte[] uv) BgrToNv12(byte[] bgr, int w, int h, int step)
    {
        var y = new byte[w * h];
        var uv = new byte[w * (h / 2)];
        for (int j = 0; j < h; j++)
        {
            int row = j * step;
            for (int i = 0; i < w; i++)
            {
                int px = row + i * 3;
                int B = bgr[px], G = bgr[px + 1], R = bgr[px + 2];
                int Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
                Y = Math.Clamp(Y, 0, 255);
                y[j * w + i] = (byte)Y;
                if ((j & 1) == 0 && (i & 1) == 0)
                {
                    int U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
                    int V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;
                    U = Math.Clamp(U, 0, 255);
                    V = Math.Clamp(V, 0, 255);
                    int o = (j / 2) * w + i;
                    uv[o] = (byte)U;
                    uv[o + 1] = (byte)V;
                }
            }
        }
        return (y, uv);
    }

    // JPEG -> BGR（BMP 自底向上需翻转行）-> NV12 host 图
    private static (VisionImage img, byte[] y, byte[] uv) LoadHostNv12(VisionImage src)
    {
        int w = src.Width, h = src.Height;
        byte[] bgr = src.ToByteArray();
        for (int j = 0; j < h / 2; j++)
            for (int i = 0; i < w * 3; i++)
            {
                int a = j * w * 3 + i, b = (h - 1 - j) * w * 3 + i;
                (bgr[a], bgr[b]) = (bgr[b], bgr[a]);
            }
        var (y, uv) = BgrToNv12(bgr, w, h, w * 3);
        return (VisionImage.FromNv12Data(y, uv, w, h, w, w), y, uv);
    }

    // host NV12 -> GPU 零拷贝 device NV12 图（返回管理句柄以释放）
    private static (VisionImage img, IntPtr dY, IntPtr dUv) LoadDeviceNv12(byte[] y, byte[] uv, int w, int h)
    {
        if (Nv12Cudart.cudaMalloc(out var dY, (ulong)(w * h)) != 0 ||
            Nv12Cudart.cudaMalloc(out var dUv, (ulong)(w * h / 2)) != 0)
            throw new InvalidOperationException("cudaMalloc failed");

        GCHandle gY = GCHandle.Alloc(y, GCHandleType.Pinned);
        GCHandle gUv = GCHandle.Alloc(uv, GCHandleType.Pinned);
        try
        {
            IntPtr py = Marshal.UnsafeAddrOfPinnedArrayElement(y, 0);
            IntPtr pu = Marshal.UnsafeAddrOfPinnedArrayElement(uv, 0);
            if (Nv12Cudart.cudaMemcpy(dY, py, (ulong)(w * h), 1) != 0 ||
                Nv12Cudart.cudaMemcpy(dUv, pu, (ulong)(w * h / 2), 1) != 0)
                throw new InvalidOperationException("cudaMemcpy H2D failed");
        }
        finally
        {
            gY.Free();
            gUv.Free();
        }
        return (VisionImage.FromDeviceNv12(dY, dUv, w, h, w, w, Device.GPU), dY, dUv);
    }

    private static PedestrianAttributeModel CreateTrtPedAttr()
    {
        var opt = new RuntimeOption();
        opt.SetDevice(Device.GPU).UseTrt();
        // 两参重载：det 模型路径 + 分类模型路径（内部自动拼成 capi 的 "det|cls"）
        return new PedestrianAttributeModel(DetEngine, ClsEngine, opt);
    }

    private static void Configure(PedestrianAttributeModel ped)
    {
        ped.SetInputSize(1280, 1280);
        ped.SetClsInputSize(192, 256);
        ped.SetClsBatchSize(8);
        ped.SetDetThreshold(0.25);
    }

    private static AttributeResult[] Predict(PedestrianAttributeModel ped, VisionImage img)
    {
        var r = ped.Predict(img);
        return r.ToArray();
    }

    [Test]
    public void PedAttr_Nv12_DeviceVsHost_ConsistentAfterHostFirst()
    {
        using var src = VisionImage.Read(TestImage);
        // host NV12
        var (hi, y, uv) = LoadHostNv12(src);
        int w = src.Width, h = src.Height;

        using var ped = CreateTrtPedAttr();
        Configure(ped);

        AttributeResult[] hostRes;
        using (hi)
        {
            hostRes = Predict(ped, hi);
        }
        Assert.That(hostRes.Length, Is.GreaterThan(0), "host NV12 should detect persons");

        // 设备：先有 host 预测后再跑 device（回归跨流同步竞争 Bug 的场景）
        var (di, dY, dUv) = LoadDeviceNv12(y, uv, w, h);
        try
        {
            using (di)
            {
                var devRes = Predict(ped, di);
                Assert.That(devRes.Length, Is.EqualTo(hostRes.Length),
                    "device NV12 detection count should match host");

                for (int i = 0; i < devRes.Length; i++)
                {
                    var a = devRes[i].AttrScores;
                    var ha = hostRes[i].AttrScores;
                    Assert.That(a.Length, Is.EqualTo(ha.Length), "attr dim mismatch");
                    for (int k = 0; k < a.Length; k++)
                        Assert.That(Math.Abs(a[k] - ha[k]), Is.LessThan(0.25f),
                            $"person[{i}] attr[{k}] device={a[k]} host={ha[k]}");
                }
            }
        }
        finally
        {
            Nv12Cudart.cudaFree(dY);
            Nv12Cudart.cudaFree(dUv);
        }
    }

    [Test]
    public void PedAttr_Nv12_Device_RepeatedCalls_Stable()
    {
        using var src = VisionImage.Read(TestImage);
        var (hi, y, uv) = LoadHostNv12(src);
        hi.Dispose();
        int w = src.Width, h = src.Height;

        using var ped = CreateTrtPedAttr();
        Configure(ped);

        var (di, dY, dUv) = LoadDeviceNv12(y, uv, w, h);
        try
        {
            AttributeResult[] first;
            using (di)
            {
                first = Predict(ped, di);
            }
            Assert.That(first.Length, Is.GreaterThan(0));

            // 重新载入同一 device 帧，再跑一次，应与第一次一致
            var (di2, dY2, dUv2) = LoadDeviceNv12(y, uv, w, h);
            try
            {
                AttributeResult[] second;
                using (di2)
                {
                    second = Predict(ped, di2);
                }
                Assert.That(second.Length, Is.EqualTo(first.Length), "count unstable across repeats");
                for (int i = 0; i < first.Length; i++)
                    for (int k = 0; k < first[i].AttrScores.Length; k++)
                        Assert.That(Math.Abs(first[i].AttrScores[k] - second[i].AttrScores[k]),
                            Is.LessThan(0.01f), $"repeat person[{i}] attr[{k}] unstable");
            }
            finally
            {
                Nv12Cudart.cudaFree(dY2);
                Nv12Cudart.cudaFree(dUv2);
            }
        }
        finally
        {
            Nv12Cudart.cudaFree(dY);
            Nv12Cudart.cudaFree(dUv);
        }
    }

    // CUDA runtime 直调（仅设备直通测试用）
    private static partial class Nv12Cudart
    {
        [DllImport("libcudart.so", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int cudaDeviceCount(out int n);
        [DllImport("libcudart.so", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int cudaMalloc(out IntPtr p, ulong sz);
        [DllImport("libcudart.so", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int cudaMemcpy(IntPtr dst, IntPtr src, ulong cnt, int kind);
        [DllImport("libcudart.so", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int cudaFree(IntPtr p);
    }
}

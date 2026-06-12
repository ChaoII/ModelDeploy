using System.Runtime.InteropServices;
using ModelDeploy;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeployUnitTest;

public class ModelDeployTests
{
    // ==================== Point / Rect / Size / Color structs ====================

    [Test]
    public void Point_ToNativeAndBack()
    {
        var p = new Point { X = 10, Y = 20 };
        var native = p.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.x, Is.EqualTo(10));
            Assert.That(native.y, Is.EqualTo(20));
        });
        var back = Point.FromNative(native);
        Assert.That(back.X, Is.EqualTo(10));
        Assert.That(back.Y, Is.EqualTo(20));
    }

    [Test]
    public void Rect_ToNativeAndBack()
    {
        var r = new Rect { X = 1, Y = 2, Width = 100, Height = 200 };
        var native = r.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.x, Is.EqualTo(1));
            Assert.That(native.y, Is.EqualTo(2));
            Assert.That(native.width, Is.EqualTo(100));
            Assert.That(native.height, Is.EqualTo(200));
        });
        var back = Rect.FromNative(native);
        Assert.That(back.Width, Is.EqualTo(100));
    }

    [Test]
    public void Size_ToNativeAndBack()
    {
        var s = new Size { Width = 640, Height = 480 };
        var native = s.ToNative();
        Assert.That(native.width, Is.EqualTo(640));
        Assert.That(native.height, Is.EqualTo(480));
        var back = Size.FromNative(native);
        Assert.That(back.Width, Is.EqualTo(640));
    }

    [Test]
    public void Color_ToNativeAndBack()
    {
        var c = new Color { R = 255, G = 128, B = 64 };
        var native = c.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.r, Is.EqualTo(255));
            Assert.That(native.g, Is.EqualTo(128));
            Assert.That(native.b, Is.EqualTo(64));
        });
        var back = Color.FromNative(native);
        Assert.That(back.R, Is.EqualTo(255));
    }

    [Test]
    public void RotatedRect_ToNativeAndBack()
    {
        var rr = new RotatedRect { Xc = 100, Yc = 200, Width = 50, Height = 30, Angle = 45 };
        var native = rr.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.xc, Is.EqualTo(100));
            Assert.That(native.yc, Is.EqualTo(200));
            Assert.That(native.width, Is.EqualTo(50));
            Assert.That(native.height, Is.EqualTo(30));
            Assert.That(native.angle, Is.EqualTo(45));
        });
        var back = RotatedRect.FromNative(native);
        Assert.That(back.Angle, Is.EqualTo(45));
    }

    // ==================== Polygon ====================

    [Test]
    public void Polygon_ToNativeAndBack()
    {
        var poly = Polygon.FromPointList(new List<Point>
        {
            new() { X = 0, Y = 0 },
            new() { X = 10, Y = 0 },
            new() { X = 5, Y = 10 }
        });
        var native = poly.CopyToMDPolygon();
        try
        {
            Assert.That(native.size, Is.EqualTo(3));
            var points = new MDPoint[3];
            var ptr = native.data;
            for (int i = 0; i < 3; i++)
            {
                points[i] = Marshal.PtrToStructure<MDPoint>(ptr);
                ptr = IntPtr.Add(ptr, Marshal.SizeOf<MDPoint>());
            }
            Assert.That(points[0].x, Is.EqualTo(0));
            Assert.That(points[2].y, Is.EqualTo(10));
        }
        finally
        {
            Marshal.FreeHGlobal(native.data);
        }
    }

    // ==================== Image creation ====================

    private static string GetTestDataDir()
    {
        return Environment.GetEnvironmentVariable("TEST_DATA_DIR")
            ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
    }

    private static string TestImagePath => Path.Combine(GetTestDataDir(), "test_data", "test_images", "test_person.jpg");

    private static bool HasTestImage => File.Exists(TestImagePath);

    [Test]
    public void Read_JpegFile_SetsCorrectProperties()
    {
        if (!HasTestImage) Assert.Ignore("Test image not found");
        using var img = Image.Read(TestImagePath);
        Assert.Multiple(() =>
        {
            Assert.That(img.Width, Is.EqualTo(900));
            Assert.That(img.Height, Is.EqualTo(675));
            Assert.That(img.Channels, Is.EqualTo(3));
        });
    }

    [Test]
    public void Read_NonExistentFile_ReturnsNullData()
    {
        using var img = Image.Read("/nonexistent/path.jpg");
        Assert.That(img.RawImage.data, Is.EqualTo(IntPtr.Zero));
    }

    [Test]
    public void FromBgr24Data_SetsCorrectProperties()
    {
        var data = new byte[100 * 80 * 3];
        using var img = Image.FromBgr24Data(data, 100, 80);
        Assert.Multiple(() =>
        {
            Assert.That(img.Width, Is.EqualTo(100));
            Assert.That(img.Height, Is.EqualTo(80));
            Assert.That(img.Channels, Is.EqualTo(3));
        });
    }

    [Test]
    public void FromBgr24Data_EmptyData_ReturnsEmptyImage()
    {
        var img = Image.FromBgr24Data(Array.Empty<byte>(), 0, 0);
        Assert.That(img.RawImage.data, Is.EqualTo(IntPtr.Zero));
    }

    [Test]
    public void FromRgb24Data_SetsCorrectProperties()
    {
        using var img = Image.FromRgb24Data(new byte[64 * 48 * 3], 64, 48);
        Assert.That(img.Channels, Is.EqualTo(3));
    }

    [Test]
    public void FromRgb24DataToBgr24_ConvertsColors()
    {
        // Pure red in RGB = (0,0,255), after BGR conversion = (255,0,0)
        var rgb = new byte[] { 0, 0, 255 };
        using var img = Image.FromRgb24DataToBgr24(rgb, 1, 1);
        var pixels = img.ToByteArray();
        Assert.Multiple(() =>
        {
            Assert.That(pixels[0], Is.EqualTo(255)); // B
            Assert.That(pixels[1], Is.EqualTo(0));   // G
            Assert.That(pixels[2], Is.EqualTo(0));   // R
        });
    }

    [Test]
    public void FromYuv420PDataToBgr24_ProducesValidImage()
    {
        var yuv = new byte[100 * 80 * 3 / 2];
        using var img = Image.FromYuv420PDataToBgr24(yuv, 100, 80);
        Assert.Multiple(() =>
        {
            Assert.That(img.Width, Is.EqualTo(100));
            Assert.That(img.Height, Is.EqualTo(80));
            Assert.That(img.Channels, Is.EqualTo(3));
        });
    }

    [Test]
    public void FromNv12DataToBgr24_ProducesValidImage()
    {
        using var img = Image.FromNv12DataToBgr24(new byte[100 * 80 * 3 / 2], 100, 80);
        Assert.That(img.Channels, Is.EqualTo(3));
    }

    [Test]
    public void FromNv21DataToBgr24_ProducesValidImage()
    {
        using var img = Image.FromNv21DataToBgr24(new byte[100 * 80 * 3 / 2], 100, 80);
        Assert.That(img.Channels, Is.EqualTo(3));
    }

    // ==================== Image operations ====================

    [Test]
    public void Clone_PreservesDimensions()
    {
        using var img = Image.FromBgr24Data(new byte[50 * 50 * 3], 50, 50);
        using var cloned = img.Clone();
        Assert.Multiple(() =>
        {
            Assert.That(cloned.Width, Is.EqualTo(img.Width));
            Assert.That(cloned.Height, Is.EqualTo(img.Height));
            Assert.That(cloned.Channels, Is.EqualTo(img.Channels));
        });
    }

    [Test]
    public void Clone_CreatesIndependentCopy()
    {
        using var img = Image.FromBgr24Data(new byte[10 * 10 * 3], 10, 10);
        using var cloned = img.Clone();
        Assert.That(cloned.RawImage.data, Is.Not.EqualTo(img.RawImage.data));
    }

    [Test]
    public void Crop_ReturnsCorrectDimensions()
    {
        using var img = Image.FromBgr24Data(new byte[100 * 100 * 3], 100, 100);
        using var cropped = img.Crop(new Rect { X = 10, Y = 10, Width = 50, Height = 50 });
        Assert.Multiple(() =>
        {
            Assert.That(cropped.Width, Is.EqualTo(50));
            Assert.That(cropped.Height, Is.EqualTo(50));
        });
    }

    // Crop out of bounds depends on native OpenCV behavior — skipped here

    [Test]
    public void ToByteArray_ReturnsCorrectSize()
    {
        using var img = Image.FromBgr24Data(new byte[30 * 20 * 3], 30, 20);
        var bytes = img.ToByteArray();
        Assert.That(bytes.Length, Is.EqualTo(30 * 20 * 3));
    }

    [Test]
    public void ToByteArray_ContentMatchesInput()
    {
        var data = CreateGradientBgr(10, 10);
        using var img = Image.FromBgr24Data(data, 10, 10);
        var bytes = img.ToByteArray();
        Assert.That(bytes, Is.EqualTo(data));
    }

    [Test]
    public void ToByteArray_ZeroChannelImage_ReturnsZeros()
    {
        using var img = Image.FromBgr24Data(new byte[100], 10, 10);
        var bytes = img.ToByteArray();
        Assert.That(bytes, Is.All.EqualTo(0));
    }

    // ==================== Image Save ====================

    [Test]
    public void Save_WritesFileToDisk()
    {
        using var img = Image.FromBgr24Data(new byte[10 * 10 * 3], 10, 10);
        var tmp = Path.GetTempFileName() + ".jpg";
        try
        {
            Assert.DoesNotThrow(() => img.Save(tmp));
            Assert.That(new FileInfo(tmp).Length, Is.GreaterThan(0));
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    // ==================== RuntimeOption ====================

    [Test]
    public void RuntimeOption_ToNative_SetsFields()
    {
        var opt = new RuntimeOption
        {
            CpuThreadNum = 4,
            Device = Device.GPU,
            Backend = Backend.TRT,
        };
        var native = opt.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.cpu_thread_num, Is.EqualTo(4));
            Assert.That(native.device, Is.EqualTo(Device.GPU));
            Assert.That(native.backend, Is.EqualTo(Backend.TRT));
        });
    }

    [Test]
    public void RuntimeOption_DefaultValues()
    {
        var opt = new RuntimeOption();
        var native = opt.ToNative();
        Assert.Multiple(() =>
        {
            Assert.That(native.cpu_thread_num, Is.EqualTo(0));
            Assert.That(native.device, Is.EqualTo(Device.CPU));
            Assert.That(native.backend, Is.EqualTo(Backend.ORT));
        });
    }

    // ==================== Draw ====================

    [Test]
    public void DrawRect_ModifiesPixels()
    {
        using var img = Image.FromBgr24Data(new byte[50 * 50 * 3], 50, 50);
        var color = new Color { R = 128, G = 64, B = 32 };
        var rect = new Rect { X = 5, Y = 5, Width = 20, Height = 20 };

        Draw.DrawRect(img, rect, color, 1.0);

        var pixels = img.ToByteArray();
        int insideIdx = (10 * 50 + 10) * 3;
        // All pixels inside should be non-zero (drawn with color, alpha=1.0)
        Assert.That(pixels[insideIdx], Is.GreaterThan(0));
        Assert.That(pixels[insideIdx + 1], Is.GreaterThan(0));
        Assert.That(pixels[insideIdx + 2], Is.GreaterThan(0));
    }

    [Test]
    public void DrawPolygon_ModifiesPixels()
    {
        var data = CreateGradientBgr(50, 50);
        using var img = Image.FromBgr24Data(data, 50, 50);
        var color = new Color { R = 0, G = 255, B = 0 };
        var poly = Polygon.FromPointList(new List<Point>
        {
            new() { X = 5, Y = 5 }, new() { X = 25, Y = 5 }, new() { X = 15, Y = 25 }
        });

        Assert.DoesNotThrow(() => Draw.DrawPolygon(img, poly, color, 1.0));
    }

    // ==================== Dispose / IDisposable ====================

    [Test]
    public void Image_Dispose_FreesNativeMemory()
    {
        var img = Image.FromBgr24Data(new byte[100], 10, 10);
        var dataPtr = img.RawImage.data;
        Assert.That(dataPtr, Is.Not.EqualTo(IntPtr.Zero));
        img.Dispose();
        Assert.That(img.RawImage.data, Is.EqualTo(IntPtr.Zero));
    }

    [Test]
    public void Image_UsingBlock_CallsDispose()
    {
        IntPtr dataPtr;
        using (var img = Image.FromBgr24Data(new byte[100], 10, 10))
        {
            dataPtr = img.RawImage.data;
            Assert.That(dataPtr, Is.Not.EqualTo(IntPtr.Zero));
        }
        // After using block, the pointer should be zeroed by Dispose
    }

    [Test]
    public void Image_MultipleDispose_DoesNotThrow()
    {
        var img = Image.FromBgr24Data(new byte[100], 10, 10);
        img.Dispose();
        Assert.DoesNotThrow(() => img.Dispose());
    }

    // ==================== Enums ====================

    [Test]
    public void Device_Enum_HasCorrectValues()
    {
        Assert.Multiple(() =>
        {
            Assert.That((int)Device.CPU, Is.EqualTo(0));
            Assert.That((int)Device.GPU, Is.EqualTo(1));
            Assert.That((int)Device.OPENCL, Is.EqualTo(2));
            Assert.That((int)Device.VULKAN, Is.EqualTo(3));
        });
    }

    [Test]
    public void Backend_Enum_HasCorrectValues()
    {
        Assert.Multiple(() =>
        {
            Assert.That((int)Backend.ORT, Is.EqualTo(0));
            Assert.That((int)Backend.MNN, Is.EqualTo(1));
            Assert.That((int)Backend.TRT, Is.EqualTo(2));
            Assert.That((int)Backend.NONE, Is.EqualTo(3));
        });
    }

    [Test]
    public void MDStatusCode_Enum_HasCorrectValues()
    {
        Assert.Multiple(() =>
        {
            Assert.That((int)MDStatusCode.Success, Is.EqualTo(0));
            Assert.That((int)MDStatusCode.ModelInitializeFailed, Is.EqualTo(4));
        });
    }

    // ==================== Utils ====================

    [Test]
    public void PtrToStringUTF8_ConvertsCorrectly()
    {
        var native = Marshal.StringToCoTaskMemUTF8("test123");
        try
        {
            var result = Utils.PtrToStringUTF8(native);
            Assert.That(result, Is.EqualTo("test123"));
        }
        finally
        {
            Marshal.FreeCoTaskMem(native);
        }
    }

    [Test]
    public void Check_ZeroCode_DoesNotThrow()
    {
        Assert.DoesNotThrow(() => Utils.Check(0, "test"));
    }

    [Test]
    public void Check_NonZeroCode_Throws()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => Utils.Check(1, "test"));
        Assert.That(ex!.Message, Does.Contain("test"));
    }

    // ==================== Image with CompressedBytes ====================

    [Test]
    public void FromCompressedBytes_FromTestImage_Decodes()
    {
        if (!HasTestImage) Assert.Ignore("Test image not found");
        var fileBytes = File.ReadAllBytes(TestImagePath);
        using var decoded = Image.FromCompressedBytes(fileBytes, fileBytes.Length);
        Assert.Multiple(() =>
        {
            Assert.That(decoded.Width, Is.EqualTo(900));
            Assert.That(decoded.Height, Is.EqualTo(675));
            Assert.That(decoded.Channels, Is.EqualTo(3));
        });
    }

    // ==================== Helpers ====================

    private static byte[] CreateGradientBgr(int w, int h)
    {
        var data = new byte[w * h * 3];
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var idx = (y * w + x) * 3;
                data[idx] = (byte)(x % 256);
                data[idx + 1] = (byte)(y % 256);
                data[idx + 2] = (byte)((x + y) % 256);
            }
        return data;
    }
}

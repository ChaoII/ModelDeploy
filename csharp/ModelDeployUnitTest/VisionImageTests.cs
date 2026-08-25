using System;
using System.Runtime.InteropServices;
using ModelDeploy;
using NUnit.Framework;

namespace ModelDeployUnitTest;

/// <summary>
/// 裸指针设备归属校验（VisionImage.ValidatePointerDevice）：
///   CPU 非空 → true；CPU 零指针 → InvalidOperationException；TPU → NotSupportedException。
/// </summary>
public class VisionImageTests
{
    [Test]
    public void ValidatePointerDevice_Cpu_NonNull_ReturnsTrue()
    {
        var p = Marshal.AllocHGlobal(16);
        try { Assert.That(VisionImage.ValidatePointerDevice(p, Device.CPU), Is.True); }
        finally { Marshal.FreeHGlobal(p); }
    }

    [Test]
    public void ValidatePointerDevice_Cpu_Zero_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => VisionImage.ValidatePointerDevice(IntPtr.Zero, Device.CPU));
    }

    [Test]
    public void ValidatePointerDevice_Tpu_ThrowsNotSupported()
    {
        Assert.Throws<NotSupportedException>(() => VisionImage.ValidatePointerDevice(new IntPtr(1), Device.TPU));
    }
}

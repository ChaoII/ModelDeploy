using System.Runtime.InteropServices;
using System.Text;

namespace ModelDeploy.utils;

public class Utils
{
    public static IntPtr ConvertStringToHGlobalUtf8(string str)
    {
        byte[] utf8Bytes = Encoding.UTF8.GetBytes(str);
        IntPtr unmanagedString = Marshal.AllocHGlobal(utf8Bytes.Length + 1);
        Marshal.Copy(utf8Bytes, 0, unmanagedString, utf8Bytes.Length);
        Marshal.WriteByte(unmanagedString + utf8Bytes.Length, 0);
        return unmanagedString;
    }
}
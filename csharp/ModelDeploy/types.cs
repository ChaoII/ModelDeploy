using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    public class Point
    {
        public int X { get; set; }
        public int Y { get; set; }

        public override string ToString() => $"{{x:{X}, y:{Y}}}";

        public static Point FromNative(MDPoint point)
        {
            return new Point { X = point.x, Y = point.y };
        }

        public MDPoint ToNative()
        {
            return new MDPoint { x = X, y = Y };
        }
    }

    public class Point3F
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public override string ToString() => $"{{x:{X}, y:{Y},  z:{Z}}}";

        public static Point3F FromNative(MDPoint3f point)
        {
            return new Point3F { X = point.x, Y = point.y, Z = point.z };
        }

        public MDPoint3f ToNative()
        {
            return new MDPoint3f { x = X, y = Y, z = Z };
        }
    }


    public class Color
    {
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }

        public static Color FromNative(MDColor color)
        {
            return new Color { R = color.r, G = color.g, B = color.b };
        }

        public MDColor ToNative()
        {
            return new MDColor { r = R, g = G, b = B };
        }
    }

    public class Polygon
    {
        private List<Point> Points { get; set; } = new List<Point>();

        public override string ToString() => $"{{ {string.Join(",", Points)} }}";

        public static Polygon FromPointList(List<Point> points)
        {
            return new Polygon { Points = points };
        }

        public static Polygon FromNative(MDPolygon cPolygon)
        {
            var polygon = new Polygon();
            int size = Marshal.SizeOf<MDPoint>();

            for (int i = 0; i < cPolygon.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cPolygon.data, i * size);
                var point = Marshal.PtrToStructure<MDPoint>(currentPtr);
                polygon.Points.Add(Point.FromNative(point));
            }

            return polygon;
        }

        public static List<Polygon> FromNativeArray(ref MDPolygon cPolygons, int size)
        {
            var polygons = new List<Polygon>();
            int structSize = Marshal.SizeOf<MDPolygon>();

            for (int i = 0; i < size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cPolygons.data, i * structSize);
                var cPolygon = Marshal.PtrToStructure<MDPolygon>(currentPtr);
                polygons.Add(FromNative(cPolygon));
            }

            return polygons;
        }

        public MDPolygon CopyToMDPolygon()
        {
            int pointSize = Marshal.SizeOf<MDPoint>();
            var cPolygon = new MDPolygon
            {
                size = Points.Count,
                data = Marshal.AllocHGlobal(Points.Count * pointSize)
            };

            for (int i = 0; i < Points.Count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cPolygon.data, i * pointSize);
                Marshal.StructureToPtr(Points[i].ToNative(), currentPtr, false);
            }

            return cPolygon;
        }

        public static MDPolygon[] ToMDPolygonArray(List<Polygon> polygons)
        {
            var cPolygons = new MDPolygon[polygons.Count];
            for (int i = 0; i < polygons.Count; i++)
            {
                cPolygons[i] = polygons[i].CopyToMDPolygon();
            }

            return cPolygons;
        }

        public Rect ToRect()
        {
            float minX = float.MaxValue;
            float maxX = float.MinValue;
            float minY = float.MaxValue;
            float maxY = float.MinValue;
            for (int i = 0; i < Points.Count; i++)
            {
                if (Points[i].X < minX) minX = Points[i].X;
                if (Points[i].X > maxX) maxX = Points[i].X;
                if (Points[i].Y < minY) minY = Points[i].Y;
                if (Points[i].Y > maxY) maxY = Points[i].Y;
            }

            return new Rect { X = (int)minX, Y = (int)minY, Width = (int)(maxX - minX), Height = (int)(maxY - minY) };
        }

        public static void FreeMDPolygon(MDPolygon polygon)
        {
            if (polygon.data != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(polygon.data);
                polygon.data = IntPtr.Zero;
            }
        }
    }

    public class Size
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public override string ToString() => $"Size {{ width: {Width}, height: {Height} }}";

        public static Size FromNative(MDSize size)
        {
            return new Size { Width = size.width, Height = size.height };
        }

        public MDSize ToNative()
        {
            return new MDSize { width = Width, height = Height };
        }
    }

    public class Rect
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

        public override string ToString() => $"Rect {{ x: {X}, y: {Y}, width: {Width}, height: {Height} }}";

        public static Rect FromNative(MDRect rect)
        {
            return new Rect { X = rect.x, Y = rect.y, Width = rect.width, Height = rect.height };
        }

        public MDRect ToNative()
        {
            return new MDRect { x = X, y = Y, width = Width, height = Height };
        }
    }


    public class RotatedRect
    {
        public float Xc { get; set; }
        public float Yc { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
        public float Angle { get; set; }

        public override string ToString() =>
            $"Rect {{ x: {Xc}, y: {Yc}, width: {Width}, height: {Height},  angle: {Angle} }}";

        public static RotatedRect FromNative(MDRotatedRect rect)
        {
            return new RotatedRect
                { Xc = rect.xc, Yc = rect.yc, Width = rect.width, Height = rect.height, Angle = rect.angle };
        }

        public MDRotatedRect ToNative()
        {
            return new MDRotatedRect { xc = Xc, yc = Yc, width = Width, height = Height, angle = Angle };
        }
    }


    public class Image
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int Channels { get; set; }

        public MDImage RawImage;
        
        public static Image Read(string imagePath)
        {
            var nativeImage = md_read_image(imagePath) ;
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }

        public static Image FromCompressedBytes(byte[] data, int byteSize)
        {
            var nativeImage = md_from_compressed_bytes(data, byteSize);
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }

        public static Image FromBgr24Data(byte[] data, int width, int height)
        {
            var nativeImage = md_from_bgr24_data(data, width, height);
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }

        public static Image FromRgb24Data(byte[] data, int width, int height)
        {
            var nativeImage = md_from_rgb24_data(data, width, height);
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }

        public static Image FromRgb24DataToBgr24(byte[] data, int width, int height)
        {
            var nativeImage = md_from_rgb24_data_to_bgr24(data, width, height);
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }


        public static Image FromBase64String(string base64String)
        {
            var nativeImage = md_from_base64_str(base64String);
            var image = new Image
            {
                Width = nativeImage.width,
                Height = nativeImage.height,
                Channels = nativeImage.channels,
                RawImage = nativeImage
            };
            return image;
        }

        private static Image FromNative(MDImage mdImage)
        {
            var image = new Image
            {
                Width = mdImage.width,
                Height = mdImage.height,
                Channels = mdImage.channels,
                RawImage = mdImage
            };
            return image;
        }

        ~Image()
        {
            md_free_image(ref RawImage);
        }

        public Image Clone()
        {
            var clone = md_clone_image(ref RawImage);
            return FromNative(clone);
        }

        public Image Crop(Rect rect)
        {
            var cropped = md_crop_image(ref RawImage, rect.ToNative());
            return FromNative(cropped);
        }

        public void Save(string imagePath) => md_save_image(ref RawImage, imagePath);
        public void Show() => md_show_image(ref RawImage);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_read_image(string imagePath);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_from_compressed_bytes(byte[] data, int byteSize);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_from_bgr24_data(byte[] data, int width, int height);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_from_rgb24_data(byte[] data, int width, int height);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_from_rgb24_data_to_bgr24(byte[] data, int width, int height);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_from_base64_str(string base64String);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_image(ref MDImage img);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_clone_image(ref MDImage image);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_crop_image(ref MDImage image, MDRect rect);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_show_image(ref MDImage image);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_save_image(ref MDImage image, string imagePath);
    }
}
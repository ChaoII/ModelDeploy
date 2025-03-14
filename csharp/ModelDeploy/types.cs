using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    public class Point
    {
        public int X { get; set; }
        public int Y { get; set; }


        public override string ToString()
        {
            return $"{{x:{X}, y:{Y}}}";
        }

        public static Point FromRaw(MDPoint point)
        {
            return new Point { X = point.x, Y = point.y };
        }

        public MDPoint ToRaw()
        {
            return new MDPoint { x = X, y = Y };
        }
    }

    public class PointF
    {
        public double X { get; set; }
        public double Y { get; set; }


        public static PointF FromRaw(MDPointF point)
        {
            return new PointF { X = point.x, Y = point.y };
        }

        public static MDPointF ToRaw(PointF point)
        {
            return new MDPointF { x = point.X, y = point.Y };
        }
    }

    public class Color
    {
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }


        public static Color FromRaw(MDColor color)
        {
            return new Color { R = color.r, G = color.g, B = color.b };
        }

        public MDColor ToRaw()
        {
            return new MDColor { r = R, g = G, b = B };
        }
    }

    public class Polygon
    {
        public override string ToString()
        {
            return $"{{ {string.Join(",", Points)} }}";
        }

        public List<Point> Points { get; set; }

        public static Polygon FromPointList(List<Point> points)
        {
            Polygon polygon = new Polygon { Points = points };
            return polygon;
        }

        public static Polygon FromMDPolygon(MDPolygon cPolygon)
        {
            Polygon polygon = new Polygon();
            List<Point> points = new List<Point>();
            for (int i = 0; i < cPolygon.size; i++)
            {
                var currentPtr = IntPtr.Add(cPolygon.data, i * Marshal.SizeOf<MDPoint>());
                points.Add(Point.FromRaw(Marshal.PtrToStructure<MDPoint>(currentPtr)));
            }

            polygon.Points = points;
            return polygon;
        }

        public static List<Polygon> FromMDPolygonArray(ref MDPolygon cPolygons, int size)
        {
            List<Polygon> polygons = new List<Polygon>();
            for (int i = 0; i < size; i++)
            {
                var currentPtr = IntPtr.Add(cPolygons.data, i * Marshal.SizeOf<MDPolygon>());
                polygons.Add(Polygon.FromMDPolygon(Marshal.PtrToStructure<MDPolygon>(currentPtr)));
            }

            return polygons;
        }

        public MDPolygon CopyToMDPolygon()
        {
            MDPolygon cPolygon = new MDPolygon { size = Points.Count };
            // 注意此处开辟的内存，因此需要不用时进行释放（如果单独使用CopyToMDPolygon后需要释放，如果使用后地址赋值给其他结构体，
            // 然后其他结构体调用了释放方法则需要在其他地方进行释）
            cPolygon.data = Marshal.AllocHGlobal(Points.Count * Marshal.SizeOf<MDPoint>());
            for (int i = 0; i < Points.Count; i++)
            {
                var currentPtr = IntPtr.Add(cPolygon.data, i * Marshal.SizeOf<MDPoint>());
                Marshal.StructureToPtr(Points[i].ToRaw(), currentPtr, false);
            }

            return cPolygon;
        }


        public static MDPolygon[] ToMDPolygonArray(List<Polygon> polygons)
        {
            MDPolygon[] cPolygons = new MDPolygon[polygons.Count];
            for (int i = 0; i < polygons.Count; i++)
            {
                MDPolygon polygon = polygons[i].CopyToMDPolygon();
                cPolygons[i] = polygon;
            }

            return cPolygons;
        }


        public static void FreeMDPolygon(MDPolygon polygon)
        {
            if (polygon.data != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(polygon.data);
                polygon.data = IntPtr.Zero; // 确保释放后不再使用
            }
        }


        ~Polygon()
        {
            // md_free_polygon(ref RawPolygon); // 修改：添加析构函数来释放未托管资源
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_polygon(ref MDPolygon polygon);
    }

    public class Size
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public override string ToString()
        {
            return $"Size {{ width: {Width}, height: {Height} }}";
        }

        public static Size FromRaw(MDSize size)
        {
            return new Size { Width = size.width, Height = size.height };
        }

        public MDSize ToRaw()
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

        public override string ToString()
        {
            return $"Rect {{ x: {X}, y: {Y}, width: {Width}, height: {Height} }}";
        }

        public static Rect FromRaw(MDRect rect)
        {
            return new Rect { X = rect.x, Y = rect.y, Width = rect.width, Height = rect.height };
        }

        public MDRect ToRaw()
        {
            return new MDRect { x = X, y = Y, width = Width, height = Height };
        }
    }

    public class Image
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int Channels { get; set; }

        // 持有生命周期
        public MDImage RawImage;


        public void Update()
        {
            Width = RawImage.width;
            Height = RawImage.height;
            Channels = RawImage.channels;
        }

        public static Image Read(string imagePath)
        {
            var image = new Image { RawImage = md_read_image(imagePath) };
            image.Update();
            return image;
        }

        public static Image FromRaw(MDImage mdImage)
        {
            Image image = new Image { RawImage = mdImage };
            image.Update();
            return image;
        }

        ~Image()
        {
            md_free_image(ref RawImage);
        }

        public Image Clone()
        {
            MDImage cloneImage = md_clone_image(ref RawImage);
            Image image = FromRaw(cloneImage);
            return image;
        }

        public Image Crop(Rect rect)
        {
            MDImage cropImage = md_crop_image(ref RawImage, rect.ToRaw());
            Image image = FromRaw(cropImage);
            return image;
        }

        public void Save(string imagePath)
        {
            md_save_image(ref RawImage, imagePath);
        }

        public void Show()
        {
            md_show_image(ref RawImage);
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_read_image(string imagePath);

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
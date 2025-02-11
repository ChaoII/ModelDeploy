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
        Polygon(List<Point> points)
        {
            Points = points;
        }

        public List<Point> Points { get; set; }

        public static Polygon FromRaw(MDPolygon polygon)
        {
            List<Point> points = new List<Point>();
            for (int i = 0; i < polygon.size; i++)
            {
                var currentPtr = IntPtr.Add(polygon.data, i * Marshal.SizeOf<MDPoint>());
                points.Add(Point.FromRaw(Marshal.PtrToStructure<MDPoint>(currentPtr)));
            }

            return new Polygon(points);
        }

        public MDPolygon ToRaw()
        {
            MDPolygon polygon = new MDPolygon
            {
                size = Points.Count,
                data = Marshal.AllocHGlobal(Marshal.SizeOf<MDPoint>() * Points.Count)
            };

            for (int i = 0; i < Points.Count; i++)
            {
                var currentPtr = IntPtr.Add(polygon.data, i * Marshal.SizeOf<MDPoint>());
                Marshal.StructureToPtr(Points[i].ToRaw(), currentPtr, false);
            }

            return polygon;
        }
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

        public Image Crop(MDRect rect)
        {
            MDImage cropImage = md_crop_image(ref RawImage, rect);
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


        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_read_image(string imagePath);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_image(ref MDImage img);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_clone_image(ref MDImage image);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern MDImage md_crop_image(ref MDImage image, MDRect rect);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_show_image(ref MDImage image);


        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_save_image(ref MDImage image, string imagePath);
    }
}
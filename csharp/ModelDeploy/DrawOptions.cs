using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    /// <summary>
    /// 结果可视化选项（复用 C++ vis_* 系列）。纯托管类型，无需关心原生内存布局。
    /// </summary>
    public sealed class DrawOptions
    {
        /// <summary>置信度阈值（默认 0.5）。</summary>
        public double Threshold { get; set; } = 0.5;

        /// <summary>类别名映射：label_id -> 名称（可选）。</summary>
        public Dictionary<int, string> LabelMap { get; set; } = new();

        /// <summary>字体文件路径（可选；空则用内置字体）。</summary>
        public string FontPath { get; set; } = string.Empty;

        /// <summary>字体大小（默认 14）。</summary>
        public int FontSize { get; set; } = 14;

        /// <summary>半透明混合系数（默认 0.15）。</summary>
        public double Alpha { get; set; } = 0.15;

        /// <summary>是否保存绘制结果为 vis_result.jpg。</summary>
        public bool SaveResult { get; set; } = false;

        /// <summary>异常对象索引列表（仅人员属性生效）：命中则该对象画红框，否则绿框（默认空=全绿）。</summary>
        public List<int> AbnormalIds { get; set; } = new();

        /// <summary>是否绘制属性文本（仅人员属性生效，默认 true）。</summary>
        public bool ShowAttr { get; set; } = true;

        /// <summary>转换为原生 MDDrawOptions（label_map/font 由调用方负责 FreeNative 释放）。</summary>
        internal MDDrawOptions ToNative()
        {
            var native = new MDDrawOptions
            {
                threshold = Threshold,
                font_size = FontSize,
                alpha = Alpha,
                save_result = SaveResult ? 1 : 0,
                show_attr = ShowAttr ? 1 : 0,
            };

            // 字体路径 -> UTF-8 指针
            if (!string.IsNullOrEmpty(FontPath))
                native.font_path = Utf8Helper.Alloc(FontPath);

            // LabelMap -> MDLabelItem 数组
            if (LabelMap != null && LabelMap.Count > 0)
            {
                var items = new MDLabelItem[LabelMap.Count];
                int i = 0;
                foreach (var kv in LabelMap)
                {
                    items[i].id = kv.Key;
                    items[i].name = Utf8Helper.Alloc(kv.Value);
                    i++;
                }
                var ptr = Marshal.AllocHGlobal(Marshal.SizeOf<MDLabelItem>() * items.Length);
                for (int j = 0; j < items.Length; j++)
                    Marshal.StructureToPtr(items[j],
                        IntPtr.Add(ptr, j * Marshal.SizeOf<MDLabelItem>()), false);
                native.label_map = ptr;
                native.label_map_size = new UIntPtr((uint)items.Length);
            }

            // AbnormalIds -> int[]（仅 ATTR）
            if (AbnormalIds != null && AbnormalIds.Count > 0)
            {
                var arr = AbnormalIds.ToArray();
                var ids = Marshal.AllocHGlobal(sizeof(int) * arr.Length);
                for (int j = 0; j < arr.Length; j++)
                    Marshal.WriteInt32(IntPtr.Add(ids, j * sizeof(int)), arr[j]);
                native.abnormal_ids = ids;
                native.abnormal_ids_size = new UIntPtr((uint)arr.Length);
            }

            return native;
        }

        /// <summary>释放 ToNative 分配的 label_map / font_path 内存。</summary>
        internal static void FreeNative(MDDrawOptions native)
        {
            if (native.label_map != IntPtr.Zero)
            {
                // 释放数组内每个 name 字符串
                int n = (int)native.label_map_size;
                int stride = Marshal.SizeOf<MDLabelItem>();
                for (int i = 0; i < n; i++)
                {
                    var item = Marshal.PtrToStructure<MDLabelItem>(IntPtr.Add(native.label_map, i * stride));
                    if (item.name != IntPtr.Zero) Marshal.FreeHGlobal(item.name);
                }
                Marshal.FreeHGlobal(native.label_map);
            }
            if (native.font_path != IntPtr.Zero) Marshal.FreeHGlobal(native.font_path);
            if (native.abnormal_ids != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(native.abnormal_ids);
                native.abnormal_ids = IntPtr.Zero;
            }
        }
    }
}

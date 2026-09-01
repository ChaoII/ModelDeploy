# -*- coding: utf-8 -*-
"""修复 ONNX 模型中缺失 kernel_shape 的 Conv 节点（不改任何数值语义）。

部分 SeetaFace 老导出模型（如 age_predictor.onnx）的 Conv 节点不带
kernel_shape 属性，依赖权重张量的形状推断。onnxruntime 可容忍，
但 tpu-mlir 的 OnnxConverter 转换时 KeyError: 'kernel_shape' 崩溃。
本脚本依据权重 initializer（第 2 输入）的形状补齐 kernel_shape，并
保留一份 .orig 备份。

用法: python repair_conv_kernel_shape.py <model.onnx> [--out fixed.onnx]
"""
import os
import sys
import shutil
import onnx
from onnx import helper


def main():
    src = sys.argv[1]
    out = None
    for i, a in enumerate(sys.argv[2:]):
        if a == "--out" and i + 1 < len(sys.argv[2:]):
            out = sys.argv[2 + i + 1]
    def dims(v):
        t = v.type.tensor_type
        if not t.HasField("shape"):
            return None
        return [d.dim_value for d in t.shape.dim]

    model = onnx.load(src)
    shape_of = {v.name: dims(v) for v in model.graph.value_info if dims(v)}
    init_shape = {i.name: list(i.dims) for i in model.graph.initializer}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {a.name: a for a in node.attribute}
        if "kernel_shape" in attrs:
            continue
        w_name = node.input[1]
        wshape = init_shape.get(w_name) or shape_of.get(w_name)
        if not wshape or len(wshape) < 4:
            print("  skip (unknown weight shape):", node.name)
            continue
        # Conv2D 权重 [M, C/group, kh, kw]（4 维）
        kh, kw = wshape[-2], wshape[-1]
        node.attribute.append(helper.make_attribute("kernel_shape", [kh, kw]))
        fixed += 1
    if fixed == 0:
        print("no Conv node needed repair")
        return 0
    if out is None:
        bak = src + ".orig"
        if not os.path.exists(bak):
            shutil.copyfile(src, bak)
            print("backup ->", bak)
        out = src
    onnx.save(model, out)
    print("fixed %d Conv nodes, saved -> %s" % (fixed, out))
    return 0 if fixed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

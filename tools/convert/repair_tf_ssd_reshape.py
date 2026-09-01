# -*- coding: utf-8 -*-
"""修复 TF-SSD 导出模型的"单一分辨率烘焙"固定 Reshape 常量。

症状（fas_second.onnx 等 SeetaFace 防伪模型）：输入声明为全动态维度
[-1,-1,-1,-1]，但 BoxPredictor 的 Reshape 目标却是训练时按唯一分辨率
烘焙的常量（如 [1,1083,3,1]）：网格 19×19、每格 3 类 → 只有恰好命中
烘焙栅格的输入尺寸能推理,其余任何尺寸 ORT 直接 Reshape 失败（实际连
该尺寸本身也可能因中间层栅格取整而对不齐——1917 锚数任意整数步长都无
法复现,属导出即 bug，见 tools/convert/sophgo_artifacts.md）。

修复：把 [1, A, k, 1] / [1, A, 1, k] 中锚数维 A 改为 -1（推理），保留
其余维度，使检测头随输入分辨率自适应。转换时用固定 shape 即可由
tpu-mlir/TRT 折叠出具体数值（在 300×300 时恰好还原 1917 锚）。

用法: python repair_tf_ssd_reshape.py <model.onnx> [--out fixed.onnx]
"""
import os
import sys
import shutil
import numpy as np
import onnx


def main():
    src = sys.argv[1]
    out = None
    for i, a in enumerate(sys.argv[2:]):
        if a == "--out" and i + 1 < len(sys.argv[2:]):
            out = sys.argv[2 + i + 1]

    model = onnx.load(src)
    init = {t.name: t for t in model.graph.initializer}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if len(node.input) < 2 or node.input[1] not in init:
            continue
        t = init[node.input[1]]
        arr = onnx.numpy_helper.to_array(t)
        if arr.ndim != 1 or arr.size != 4 or int(arr[0]) != 1 or int(arr[1]) <= 0:
            continue
        # 目标形如 [1, A, k, 1] / [1, A, 1, k]，仅把锚数维 A 置为 -1
        if not (int(arr[2]) == 1 or int(arr[3]) == 1):
            continue
        new_arr = np.array([1, -1, int(arr[2]), int(arr[3])], np.int64)
        t.CopyFrom(onnx.numpy_helper.from_array(new_arr, t.name))
        fixed += 1
    if fixed == 0:
        print("no baked Reshape target needed repair")
        return 1
    if out is None:
        bak = src + ".orig"
        if not os.path.exists(bak):
            shutil.copyfile(src, bak)
            print("backup ->", bak)
        out = src
    onnx.save(model, out)
    print("fixed %d Reshape targets, saved -> %s" % (fixed, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

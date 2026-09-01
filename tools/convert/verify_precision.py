# -*- coding: utf-8 -*-
# bmodel int8 精度验证:宿主 onnxruntime(CPU)参考 vs 容器 model_runner(cmodel bmodel)。
# 输出 tools/convert/bmodel_precision_report.md(cosine / max-abs-err / 置信度是否塌)。
import json, os, sys, subprocess, glob

ROOT = r"E:\CLionProjects\ModelDeploy"
DATA = os.path.join(ROOT, "test_data", "test_models")
IMGS = os.path.join(ROOT, "test_data", "test_images")
PREC = os.path.join(ROOT, "tools", "convert", "_prec")
REG = json.load(open(os.path.join(ROOT, "tools", "convert", "models.json"), encoding="utf-8"))

def resolve(rel):
    return os.path.join(DATA, rel).replace("\\", "/") if rel else ""

def pick_img():
    for n in ("test_detection0.jpg", "test_person.jpg", "best_0.jpg", "test_ocr.png", "test_face_detection.jpg", "test_obb1.jpg"):
        p = os.path.join(IMGS, n)
        if os.path.exists(p):
            return p
    raise SystemExit("no cali image")

def main():
    import numpy as np
    import onnxruntime as ort

    os.makedirs(PREC, exist_ok=True)
    rows = []
    for m in REG:
        rel = os.path.join(m["dir"], m["stem"]) if m["dir"] else m["stem"]
        suf = "-f16" if m.get("quantize", "INT8") == "F16" else "-int8"
        bmodel = os.path.join(DATA, "sophgo", rel + suf + ".bmodel")
        onnx_p = os.path.join(DATA, "onnx", rel + ".onnx")
        if not os.path.exists(bmodel) or not os.path.exists(onnx_p):
            rows.append((m["key"], "SKIP", "", "")); continue
        shape = [int(x) for x in m["shape"].split("x")]
        iname = m["input"]
        img_p = pick_img()
        cv2 = __import__("cv2")
        im = cv2.imread(img_p)                     # BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # 与 convert.sh 校准一致(RGB)
        h, w = shape[2], shape[3]
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = im.transpose(2, 0, 1)[None].astype(np.float32)   # [1,3,H,W]

        kdir = os.path.join(PREC, m["key"]); os.makedirs(kdir, exist_ok=True)
        inp_npz = os.path.join(kdir, "input.npz").replace("\\", "/")
        np.savez(inp_npz, **{iname: x})

        # ONNX 参考输出
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = ort.InferenceSession(onnx_p, so, providers=["CPUExecutionProvider"])
        refs = sess.run(None, {iname: x})
        model_refs = [np.asarray(o).reshape(-1) for o in refs]

        # model_runner(cmodel)
        runner = "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/tools/model_runner.py"
        out_npz = (os.path.join(kdir, "out.npz")).replace("\\", "/")
        rel_slash = rel.replace("\\", "/")   # 容器内必须用正斜杠路径
        try:
            pc = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{DATA}:/conv", "-v", f"{PREC}:/tprec",
                 "tpuc_dev:1.27-slim", "python3", runner,
                 "--input", f"/tprec/{m['key']}/input.npz",
                 "--model", f"/conv/sophgo/{rel_slash}{suf}.bmodel",
                 "--output", f"/tprec/{m['key']}/out.npz"],
                check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "").strip().splitlines()[-3:]
            rows.append((m["key"], "CMODEL-ERR",
                         "rc=%d %s" % (e.returncode, " | ".join(tail)), ""))
            continue
        zoo = np.load(out_npz)
        keys = list(zoo.keys())
        # bmodel 输出按 npz key(模型输出名/顺序)对齐
        outs = [np.asarray(zoo[k]).reshape(-1) for k in keys]

        # 逐输出对齐:模型输出数>=bmodel 输出数时按序比较
        n = min(len(model_refs), len(outs))
        cos, err = [], []
        for i in range(n):
            a, b = model_refs[i].astype(np.float64), outs[i].astype(np.float64)
            if a.size == 0 or b.size == 0:
                cos.append(float("nan")); err.append(float("nan")); continue
            co = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            cos.append(co); err.append(float(np.max(np.abs(a - b))))
        maxv = float(np.max(np.abs(model_refs[0]))) if len(model_refs) else 0.0
        bmaxv = float(np.max(np.abs(outs[0]))) if len(outs) else 0.0
        rows.append((m["key"], "OK", f"cos={min(cos):.4f} max={max(err):.4g}",
                     f"onnx_max={maxv:.3g} bmodel_max={bmaxv:.3g}"))

    lines = ["# bmodel 精度验证(cmodel vs ONNX, int8/qtable 混合精度/F16)", "",
             "| key | 状态 | cos/max-err | 输出量纲 |","|---|---|---|---|"]
    for k, st, m, d in rows:
        lines.append(f"| {k} | {st} | {m} | {d} |")
    rep = os.path.join(ROOT, "tools", "convert", "bmodel_precision_report.md")
    open(rep, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()

#!/bin/bash
# 容器内批量 bmodel int8 转换(挂在 /conv=test_models, /cali_img=test_images,
# /tconf=tools/convert, /tc=tools/docker/sophgo)。用法: bash /tconf/conv_in_docker.sh [key 过滤]
set -e
FILTER="$1"
python3 - "$FILTER" <<'PYEOF'
import json, sys, subprocess, os
filter_key = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
failed = []
reg = json.load(open("/tconf/models.json"))
for m in reg:
    if filter_key and m["key"] != filter_key:
        continue
    rel = os.path.join(m["dir"], m["stem"]) if m["dir"] else m["stem"]
    shape = "[[%s]]" % m["shape"].replace("x", ",")
    quant = m.get("quantize", "INT8")
    suf = "-f16" if quant == "F16" else "-int8"
    out = "/conv/sophgo/%s%s.bmodel" % (rel, suf)
    onnx = "/conv/onnx/%s.onnx" % rel
    if not os.path.exists(onnx):
        print("skip(missing):", onnx); continue
    if os.path.exists(out):
        print("skip(exists):", out); continue
    dst_dir = os.path.dirname(out)
    os.makedirs(dst_dir, exist_ok=True)
    print("=== BMODEL:", m["key"], rel, shape, quant)
    args = ["bash", "/tc/convert.sh", "--onnx", onnx, "--name", m["stem"],
            "--shapes", shape, "--chip", "bm1688", "--quantize", quant,
            "--out", out]
    if quant == "INT8":
        args += ["--cali_images", "/cali_img", "--cali_num", "100"]
        if m.get("qtable"):
            args += ["--qtable", "/tconf/%s" % m["qtable"]]
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        failed.append((m["key"], rel, str(e)))
        print("FAILED:", m["key"], rel, "retcode=", e.returncode)
        continue
    assert os.path.exists(out), "产物缺失: %s" % out
print("BMODEL BATCH DONE, failed=%d" % len(failed))
for k, rel, err in failed:
    print("FAILED:", k, rel)
PYEOF

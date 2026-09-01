# -*- coding: utf-8 -*-
"""从 tpu-mlir 的 ORIGIN mlir 提取输出头 op(按 loc 名正则匹配), 生成混合精度 qtable。

用法: python make_qtable.py <origin.mlir> <out.qtable> <regex1> [regex2 ...]
  仅把 loc 名匹配任一正则的 top.* op 置为 F16(其余保持 INT8)。
"""
import re
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    pats = [re.compile(p) for p in sys.argv[3:]]
    locmap = {}
    for line in open(src, encoding="utf-8", errors="replace"):
        m = re.match(r'\s*#loc(\d+)\s*=\s*loc\(\s*"([^"]*)"', line)
        if m:
            locmap[m.group(1)] = m.group(2)
    names = []
    for m in re.finditer(r'"top\.[A-Za-z0-9_]+"\([^\n]*?>\s*[^\n]*?loc\(#loc(\d+)\)', open(src, encoding="utf-8", errors="replace").read()):
        nm = locmap.get(m.group(1), "")
        if nm not in names and nm:
            names.append(nm)
    if not pats:
        print("no patterns given", file=sys.stderr)
        return 1
    sel = [n for n in names if any(p.search(n) for p in pats)]
    if pats and all(p.pattern.startswith("TAIL:") for p in pats):
        frac = float(pats[0].pattern[5:])
        sel = names[int(len(names) * (1 - frac)):]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# shape-pattern qtable (auto): output head ops to F16\n")
        for n in sel:
            f.write("%s F16\n" % n)
    print("matched %d/%d ops -> %s" % (len(sel), len(names), out))
    for n in sel[:40]:
        print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

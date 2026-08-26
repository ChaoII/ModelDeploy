#!/usr/bin/env python3
"""生成内置 CJK 位图字库表 cjk_font.h(开发期,输入 ttf,输出常量数据段)。"""
import sys
from PIL import Image, ImageDraw, ImageFont

SIZE = 16


def glyph_to_bits(im: Image.Image) -> list:
    im = im.convert("L")
    bits = []
    for y in range(SIZE):
        row = 0
        for x in range(SIZE):
            if im.getpixel((x, y)) < 128:
                row |= (1 << (7 - x % 8))
        bits.append(row)
    return bits


def main(font_path: str, out_header: str) -> None:
    font = ImageFont.truetype(font_path, SIZE)
    chars = []
    for cp in range(0x20, 0x7F):
        chars.append(chr(cp))
    for zone in range(16, 56):
        qb = zone + 0xA0
        for pos in range(1, 95):
            wb = pos + 0xA0
            try:
                ch = bytes([qb, wb]).decode("gb2312")
            except UnicodeDecodeError:
                continue
            chars.append(ch)
    index = []
    glyphs = []
    for ch in chars:
        cp = ord(ch)
        im = Image.new("L", (SIZE * 2, SIZE * 2), 255)
        d = ImageDraw.Draw(im)
        d.text((0, 0), ch, font=font, fill=0)
        bbox = im.getbbox()
        if not bbox:
            continue
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = max(0, (SIZE - gw) // 2 - bbox[0])
        oy = max(0, (SIZE - gh) // 2 - bbox[1])
        cell = Image.new("L", (SIZE, SIZE), 255)
        cell.paste(im.crop(bbox), (ox, oy))
        glyphs.extend(glyph_to_bits(cell))
        index.append(cp)
    order = sorted(range(len(index)), key=lambda i: index[i])
    index = [index[i] for i in order]
    new_glyphs = []
    for i, src in enumerate(order):
        new_glyphs.append(glyphs[src * SIZE:(src + 1) * SIZE])
    flat = [b for g in new_glyphs for b in g]
    lines = []
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append(f"// 由 tools/gen_cjk_font.py 从 {font_path} 生成;共 {len(index)} 字,每字形 {SIZE}x{SIZE} 位图")
    lines.append("namespace modeldeploy::vision {")
    lines.append(f"inline const uint32_t kCjkFontIndex[] = {{{','.join(str(c) for c in index)}}};")
    lines.append(f"inline const size_t kCjkFontIndexSize = {len(index)};")
    lines.append(f"inline const uint8_t kCjkFontGlyphBytes = {SIZE};")
    lines.append(f"inline const uint8_t kCjkFontGlyphBitmaps[] = {{{','.join(str(b) for b in flat)}}};")
    lines.append("}  // namespace modeldeploy::vision")
    with open(out_header, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_header}: {len(index)} glyphs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

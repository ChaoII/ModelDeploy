#pragma once
#include "cjk_font.h"
#include <cstdint>
#include <cstddef>

namespace modeldeploy::vision {
    // 码点 → 字形下标(二分);未命中返回 -1
    inline int cjk_lookup(uint32_t cp) {
        int lo = 0, hi = static_cast<int>(kCjkFontIndexSize) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint32_t v = kCjkFontIndex[mid];
            if (v == cp) return mid;
            if (v < cp) lo = mid + 1; else hi = mid - 1;
        }
        return -1;
    }
    // 从 UTF-8 读一个码点;成功返回字节数, *cp 出码点;非法返回 0。
    // 多字节序列逐个校验续字节（bytes 非空且为合法续字节 (b & 0xC0) == 0x80）；
    // 任一续字节缺失/非法即返回 0（调用方视为无效跳过一个字节）。
    // 缓冲区 NUL 结尾，故一旦遇到 NUL（=0，非 0x80 系列）就立即返回，绝不越过终止符读取。
    inline int utf8_to_cp(const char* s, uint32_t* cp) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c < 0x80) { *cp = c; return 1; }
        if ((c & 0xE0) == 0xC0) {
            const unsigned char b1 = static_cast<unsigned char>(s[1]);
            if ((b1 & 0xC0) != 0x80) { *cp = 0; return 0; }
            *cp = ((c & 0x1F) << 6) | (b1 & 0x3F);
            return 2;
        }
        if ((c & 0xF0) == 0xE0) {
            const unsigned char b1 = static_cast<unsigned char>(s[1]);
            if ((b1 & 0xC0) != 0x80) { *cp = 0; return 0; }
            const unsigned char b2 = static_cast<unsigned char>(s[2]);
            if ((b2 & 0xC0) != 0x80) { *cp = 0; return 0; }
            *cp = ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
            return 3;
        }
        if ((c & 0xF8) == 0xF0) {
            const unsigned char b1 = static_cast<unsigned char>(s[1]);
            if ((b1 & 0xC0) != 0x80) { *cp = 0; return 0; }
            const unsigned char b2 = static_cast<unsigned char>(s[2]);
            if ((b2 & 0xC0) != 0x80) { *cp = 0; return 0; }
            const unsigned char b3 = static_cast<unsigned char>(s[3]);
            if ((b3 & 0xC0) != 0x80) { *cp = 0; return 0; }
            *cp = ((c & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
            return 4;
        }
        *cp = c; return 1;
    }
}

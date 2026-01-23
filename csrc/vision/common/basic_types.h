//
// Created by aichao on 2026/1/19.
//

#pragma once

enum class MdImageType {
    GRAY_U8 = 0, // 单通道，unsigned char存储
    GRAY_U16, // 单通道，unsigned short存储
    GRAY_S16, // 单通道，signed short存储
    GRAY_S32, // 单通道，int32存储
    GRAY_F32, // 单通道，float32存储
    GRAY_F64, // 单通道，double存储
    PLA_BGR_U8 = 20, // 三通道，unsigned char存储，存储顺序：BBB...GGG...RRR...
    PLA_RGB_U8, // 三通道，unsigned char存储，存储顺序：RRR...GGG...BBB...
    PKG_BGR_U8, // 三通道，unsigned char存储，存储顺序：BGRBGR...
    PKG_RGB_U8, // 三通道，unsigned char存储，存储顺序：RGBRGB...
    PLA_BGRA_U8, // 四通道，unsigned char存储，存储顺序：BBB...GGG...RRR...AAA...
    PLA_RGBA_U8, // 四通道，unsigned char存储，存储顺序：RRR...GGG...BBB...AAA...
    PKG_BGRA_U8, // 四通道，unsigned char存储，存储顺序：BGRABGRA...
    PKG_RGBA_U8, // 四通道，unsigned char存储，存储顺序：RGBARGBA...
    PLA_BGR_F32 = 40, // 三通道，float存储，存储顺序：BBB...GGG...RRR...
    PLA_RGB_F32, // 三通道，float存储，存储顺序：RRR...GGG...BBB...
    PKG_BGR_F32, // 三通道，float存储，存储顺序：BGRBGR...
    PKG_RGB_F32, // 三通道，float存储，存储顺序：RGBRGB...
    PLA_BGRA_F32, // 四通道，float存储，存储顺序：BBB...GGG...RRR...AAA...
    PLA_RGBA_F32, // 四通道，float存储，存储顺序：RRR...GGG...BBB...AAA...
    PKG_BGRA_F32, // 四通道，float存储，存储顺序：BGRABGRA...
    PKG_RGBA_F32, // 四通道，float存储，存储顺序：RGBARGBA...
    PKG_BGR_F64, // 三通道，double存储，存储顺序：BGRBGR...
    PKG_RGB_F64, // 三通道，double存储，存储顺序：RGBRGB...
    PKG_BGRA_F64, // 四通道，double存储，存储顺序：BGRABGRA...
    PKG_RGBA_F64, // 四通道，double存储，存储顺序：RGBARGBA...
    PKG_BGR565_U8, // 三通道，unsigned char存储，存储顺序：BGRBGR...
    PKG_RGB565_U8, // 三通道，unsigned char存储，存储顺序：RGBRGB...
    NV12 = 60, // YUV420SP类型，像素占比为Y:V:U=4:1:1，存储顺序：YYY...UVUV...
    NV21, // YVU420SP类型，像素占比为Y:U:V=4:1:1，存储顺序：YYY...VUVU...
    I420, // YUV420P类型，像素占比为Y:U:V=4:1:1，存储顺序：YYY...UUU...VVV...
    UNKNOWN
};

enum class ColorConvertType {
    CVT_PA_BGR2GRAY = 0,
    CVT_PA_RGB2GRAY,

    CVT_PA_BGR2PA_RGB,
    CVT_PA_RGB2PA_BGR,

    CVT_PA_BGR2PA_BGRA,
    CVT_PA_RGB2PA_RGBA,
    CVT_PA_BGR2PA_RGBA,
    CVT_PA_RGB2PA_BGRA,
    CVT_PA_BGRA2PA_BGR,
    CVT_PA_RGBA2PA_RGB,
    CVT_PA_RGBA2PA_BGR,
    CVT_PA_BGRA2PA_RGB,
    CVT_PA_BGRA2PA_RGBA,
    CVT_PA_RGBA2PA_BGRA,

    CVT_GRAY2PA_RGB,
    CVT_GRAY2PA_BGR,
    CVT_GRAY2PA_BGRA,
    CVT_GRAY2PA_RGBA,

    CVT_PA_BGR2PAI420,
    CVT_PA_RGB2PAI420,
    CVT_PA_BGRA2PAI420,
    CVT_PA_RGBA2PAI420,


    CVT_NV122GRAY,
    CVT_NV212GRAY,
    CVT_I4202GRAY,


    CVT_NV122PA_RGB,
    CVT_NV212PA_RGB,
    CVT_NV122PA_BGR,
    CVT_NV212PA_BGR,
    CVT_I4202PA_BGR,
    CVT_I4202PA_RGB,

    CVT_NV122PA_BGRA,
    CVT_NV212PA_BGRA,
    CVT_NV122PA_RGBA,
    CVT_NV212PA_RGBA,
    CVT_I4202PA_BGRA,
    CVT_I4202PA_RGBA,

    CVT_PA_BGR2PL_BGR,
    CVT_PA_RGB2PL_RGB,
    CVT_PL_BGR2PA_BGR,
    CVT_PL_RGB2PA_RGB,
};

enum RotateFlags {
    ROTATE_90 = 0,
    ROTATE_180 = 1,
    ROTATE_270 = 2,
};

inline std::string md_image_type_to_string(MdImageType type) {
    switch (type) {
    case MdImageType::GRAY_U8: return "GRAY_U8";
    case MdImageType::GRAY_U16: return "GRAY_U16";
    case MdImageType::GRAY_S16: return "GRAY_S16";
    case MdImageType::GRAY_S32: return "GRAY_S32";
    case MdImageType::GRAY_F32: return "GRAY_F32";
    case MdImageType::GRAY_F64: return "GRAY_F64";

    case MdImageType::PLA_BGR_U8: return "PLA_BGR_U8";
    case MdImageType::PLA_RGB_U8: return "PLA_RGB_U8";
    case MdImageType::PKG_BGR_U8: return "PKG_BGR_U8";
    case MdImageType::PKG_RGB_U8: return "PKG_RGB_U8";
    case MdImageType::PLA_BGRA_U8: return "PLA_BGRA_U8";
    case MdImageType::PLA_RGBA_U8: return "PLA_RGBA_U8";
    case MdImageType::PKG_BGRA_U8: return "PKG_BGRA_U8";
    case MdImageType::PKG_RGBA_U8: return "PKG_RGBA_U8";

    case MdImageType::PLA_BGR_F32: return "PLA_BGR_F32";
    case MdImageType::PLA_RGB_F32: return "PLA_RGB_F32";
    case MdImageType::PKG_BGR_F32: return "PKG_BGR_F32";
    case MdImageType::PKG_RGB_F32: return "PKG_RGB_F32";
    case MdImageType::PLA_BGRA_F32: return "PLA_BGRA_F32";
    case MdImageType::PLA_RGBA_F32: return "PLA_RGBA_F32";
    case MdImageType::PKG_BGRA_F32: return "PKG_BGRA_F32";
    case MdImageType::PKG_RGBA_F32: return "PKG_RGBA_F32";

    case MdImageType::PKG_BGR_F64: return "PKG_BGR_F64";
    case MdImageType::PKG_RGB_F64: return "PKG_RGB_F64";
    case MdImageType::PKG_BGRA_F64: return "PKG_BGRA_F64";
    case MdImageType::PKG_RGBA_F64: return "PKG_RGBA_F64";

    case MdImageType::PKG_BGR565_U8: return "PKG_BGR565_U8";
    case MdImageType::PKG_RGB565_U8: return "PKG_RGB565_U8";

    case MdImageType::NV12: return "NV12";
    case MdImageType::NV21: return "NV21";
    case MdImageType::I420: return "I420";
    default: return "UNKNOWN";
    }
}

inline std::string color_convert_type_to_string(ColorConvertType type) {
    switch (type) {
    case ColorConvertType::CVT_PA_BGR2GRAY: return "CVT_PA_BGR2GRAY";
    case ColorConvertType::CVT_PA_RGB2GRAY: return "CVT_PA_RGB2GRAY";

    case ColorConvertType::CVT_PA_BGR2PA_RGB: return "CVT_PA_BGR2PA_RGB";
    case ColorConvertType::CVT_PA_RGB2PA_BGR: return "CVT_PA_RGB2PA_BGR";

    case ColorConvertType::CVT_PA_BGR2PA_BGRA: return "CVT_PA_BGR2PA_BGRA";
    case ColorConvertType::CVT_PA_RGB2PA_RGBA: return "CVT_PA_RGB2PA_RGBA";
    case ColorConvertType::CVT_PA_BGR2PA_RGBA: return "CVT_PA_BGR2PA_RGBA";
    case ColorConvertType::CVT_PA_RGB2PA_BGRA: return "CVT_PA_RGB2PA_BGRA";
    case ColorConvertType::CVT_PA_BGRA2PA_BGR: return "CVT_PA_BGRA2PA_BGR";
    case ColorConvertType::CVT_PA_RGBA2PA_RGB: return "CVT_PA_RGBA2PA_RGB";
    case ColorConvertType::CVT_PA_RGBA2PA_BGR: return "CVT_PA_RGBA2PA_BGR";
    case ColorConvertType::CVT_PA_BGRA2PA_RGB: return "CVT_PA_BGRA2PA_RGB";
    case ColorConvertType::CVT_PA_BGRA2PA_RGBA: return "CVT_PA_BGRA2PA_RGBA";
    case ColorConvertType::CVT_PA_RGBA2PA_BGRA: return "CVT_PA_RGBA2PA_BGRA";

    case ColorConvertType::CVT_GRAY2PA_RGB: return "CVT_GRAY2PA_RGB";
    case ColorConvertType::CVT_GRAY2PA_BGR: return "CVT_GRAY2PA_BGR";
    case ColorConvertType::CVT_GRAY2PA_BGRA: return "CVT_GRAY2PA_BGRA";
    case ColorConvertType::CVT_GRAY2PA_RGBA: return "CVT_GRAY2PA_RGBA";

    case ColorConvertType::CVT_PA_BGR2PAI420: return "CVT_PA_BGR2PAI420";
    case ColorConvertType::CVT_PA_RGB2PAI420: return "CVT_PA_RGB2PAI420";
    case ColorConvertType::CVT_PA_BGRA2PAI420: return "CVT_PA_BGRA2PAI420";
    case ColorConvertType::CVT_PA_RGBA2PAI420: return "CVT_PA_RGBA2PAI420";

    case ColorConvertType::CVT_NV122GRAY: return "CVT_NV122GRAY";
    case ColorConvertType::CVT_NV212GRAY: return "CVT_NV212GRAY";
    case ColorConvertType::CVT_I4202GRAY: return "CVT_I4202GRAY";

    case ColorConvertType::CVT_NV122PA_RGB: return "CVT_NV122PA_RGB";
    case ColorConvertType::CVT_NV212PA_RGB: return "CVT_NV212PA_RGB";
    case ColorConvertType::CVT_NV122PA_BGR: return "CVT_NV122PA_BGR";
    case ColorConvertType::CVT_NV212PA_BGR: return "CVT_NV212PA_BGR";
    case ColorConvertType::CVT_I4202PA_BGR: return "CVT_I4202PA_BGR";
    case ColorConvertType::CVT_I4202PA_RGB: return "CVT_I4202PA_RGB";

    case ColorConvertType::CVT_NV122PA_BGRA: return "CVT_NV122PA_BGRA";
    case ColorConvertType::CVT_NV212PA_BGRA: return "CVT_NV212PA_BGRA";
    case ColorConvertType::CVT_NV122PA_RGBA: return "CVT_NV122PA_RGBA";
    case ColorConvertType::CVT_NV212PA_RGBA: return "CVT_NV212PA_RGBA";
    case ColorConvertType::CVT_I4202PA_BGRA: return "CVT_I4202PA_BGRA";
    case ColorConvertType::CVT_I4202PA_RGBA: return "CVT_I4202PA_RGBA";

    case ColorConvertType::CVT_PA_BGR2PL_BGR: return "CVT_PA_BGR2PL_BGR";
    case ColorConvertType::CVT_PA_RGB2PL_RGB: return "CVT_PA_RGB2PL_RGB";
    case ColorConvertType::CVT_PL_BGR2PA_BGR: return "CVT_PL_BGR2PA_BGR";
    case ColorConvertType::CVT_PL_RGB2PA_RGB: return "CVT_PL_RGB2PA_RGB";

    default: return "Unknown";
    }
}


inline std::ostream& operator<<(std::ostream& os, const MdImageType type) {
    return os << md_image_type_to_string(type);
}

inline std::ostream& operator<<(std::ostream& os, const ColorConvertType type) {
    return os << color_convert_type_to_string(type);
}

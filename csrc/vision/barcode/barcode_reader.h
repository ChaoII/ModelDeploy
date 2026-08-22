#pragma once
#include <cstdint>
#include <vector>
#include "vision/common/image_data.h"
#include "vision/barcode/result.h"

namespace modeldeploy::vision::barcode {
    using Formats = uint32_t;

    inline constexpr Formats FMT_QR_CODE    = 1u << 0;
    inline constexpr Formats FMT_DATA_MATRIX= 1u << 1;
    inline constexpr Formats FMT_AZTEC      = 1u << 2;
    inline constexpr Formats FMT_EAN_8      = 1u << 3;
    inline constexpr Formats FMT_EAN_13     = 1u << 4;
    inline constexpr Formats FMT_UPC_A      = 1u << 5;
    inline constexpr Formats FMT_UPC_E      = 1u << 6;
    inline constexpr Formats FMT_CODE_128   = 1u << 7;
    inline constexpr Formats FMT_CODE_39    = 1u << 8;
    inline constexpr Formats FMT_CODE_93    = 1u << 9;
    inline constexpr Formats FMT_ITF        = 1u << 10;
    inline constexpr Formats FMT_CODABAR    = 1u << 11;
    inline constexpr Formats FMT_ALL        = 0xFFFFFFFFu;

    class BarcodeReader {
    public:
        BarcodeReader() = delete;
        /*! 解码图片中的所有码。img 须为 CPU 的 packed 格式（PKG_BGR_U8 / PKG_RGB_U8 /
         *  PKG_BGRA_U8 / GRAY_U8），其它类型（NV12/NV21/I420/planar/设备类型）返回空。
         *  定义见 vision/common/image_data.h 的 ImageData（用方法 width()/height()/channels()/
         *  type()/plane(i)），并用 MdImageType 映射到 ZXing ImageFormat。 */
        static std::vector<BarcodeResult> read(const ImageData& img,
                                               Formats formats = FMT_ALL,
                                               bool convert_bgr_to_rgb = false);
    };
}

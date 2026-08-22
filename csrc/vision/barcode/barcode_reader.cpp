#include "vision/barcode/barcode_reader.h"
#include <string>
#include "ReadBarcode.h"
#include "vision/common/basic_types.h"   // MdImageType

namespace modeldeploy::vision::barcode {
    namespace {
        ZXing::BarcodeFormat to_zxing_format(Formats f) {
            switch (f) {
                case FMT_QR_CODE:     return ZXing::BarcodeFormat::QRCode;
                case FMT_DATA_MATRIX: return ZXing::BarcodeFormat::DataMatrix;
                case FMT_AZTEC:       return ZXing::BarcodeFormat::Aztec;
                case FMT_EAN_8:       return ZXing::BarcodeFormat::EAN8;
                case FMT_EAN_13:      return ZXing::BarcodeFormat::EAN13;
                case FMT_UPC_A:       return ZXing::BarcodeFormat::UPCA;
                case FMT_UPC_E:       return ZXing::BarcodeFormat::UPCE;
                case FMT_CODE_128:    return ZXing::BarcodeFormat::Code128;
                case FMT_CODE_39:     return ZXing::BarcodeFormat::Code39;
                case FMT_CODE_93:     return ZXing::BarcodeFormat::Code93;
                case FMT_ITF:         return ZXing::BarcodeFormat::ITF;
                case FMT_CODABAR:     return ZXing::BarcodeFormat::Codabar;
                default:              return ZXing::BarcodeFormat::None;
            }
        }
        // MdImageType -> ZXing ImageFormat。仅支持 CPU packed/GRAY；否则返回 None。
        ZXing::ImageFormat to_zxing_format(MdImageType t, int channels_packed) {
            (void)channels_packed;
            switch (t) {
                case MdImageType::GRAY_U8:      return ZXing::ImageFormat::Lum;
                case MdImageType::PKG_BGR_U8:   return ZXing::ImageFormat::BGR;
                case MdImageType::PKG_RGB_U8:   return ZXing::ImageFormat::RGB;
                case MdImageType::PKG_BGRA_U8:  return ZXing::ImageFormat::BGRA;
                default: return ZXing::ImageFormat::None;
            }
        }
        std::string format_name(ZXing::BarcodeFormat f) {
            return std::string(ZXing::ToString(f));
        }
        bool is_qr_format(ZXing::BarcodeFormat f) {
            return f == ZXing::BarcodeFormat::QRCode || f == ZXing::BarcodeFormat::MicroQRCode;
        }
        Point2f to_pt(const ZXing::PointI& p) { return {static_cast<float>(p.x), static_cast<float>(p.y)}; }
    }

    std::vector<BarcodeResult> BarcodeReader::read(const ImageData& img,
                                                   Formats formats,
                                                   bool convert_bgr_to_rgb) {
        std::vector<BarcodeResult> out;
        (void)convert_bgr_to_rgb;   // 兼容参数：v3 直接原生支持 BGR，无需手动转换
        if (img.empty() || img.device() != Device::CPU) return out;
        auto plane = img.plane(0);
        if (!plane.data) return out;

        ZXing::ImageFormat zfmt = to_zxing_format(img.type(), img.channels());
        if (zfmt == ZXing::ImageFormat::None) return out;   // 不支持的布局

        // 组装 BarcodeFormats 集合（该版本无 operator|=，先收集 vector 再构造）
        std::vector<ZXing::BarcodeFormat> zf_list;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            if (formats & (1u << bit)) {
                auto f = to_zxing_format(1u << bit);
                if (f != ZXing::BarcodeFormat::None) zf_list.push_back(f);
            }
        }
        ZXing::ReaderOptions options;
        options.setFormats(ZXing::BarcodeFormats(std::move(zf_list))).setTryHarder(true);

        int row_stride = plane.step > 0 ? plane.step : img.width() * img.channels();
        ZXing::ImageView view(plane.data, img.width(), img.height(), zfmt, row_stride);

        auto results = ZXing::ReadBarcodes(view, options);
        for (auto& r : results) {
            if (!r.isValid()) continue;
            BarcodeResult br;
            br.text = r.text();
            br.format = format_name(r.format());
            br.is_qr = is_qr_format(r.format());
            br.score = 1.0f;
            auto pos = r.position();
            br.quad[0] = to_pt(pos.topLeft());
            br.quad[1] = to_pt(pos.topRight());
            br.quad[2] = to_pt(pos.bottomRight());
            br.quad[3] = to_pt(pos.bottomLeft());
            out.push_back(br);
        }
        return out;
    }
}

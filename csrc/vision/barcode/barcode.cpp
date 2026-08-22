#include "vision/barcode/barcode.h"

namespace modeldeploy::vision::barcode {
    void BarcodeDetector::set_formats(Formats formats) { formats_ = formats; }

    std::vector<BarcodeResult> BarcodeDetector::detect(const ImageData& img) const {
        return BarcodeReader::read(img, formats_, /*convert_bgr_to_rgb=*/false);
    }
}

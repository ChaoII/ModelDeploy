#pragma once
#include <vector>
#include "vision/common/image_data.h"
#include "vision/barcode/result.h"
#include "vision/barcode/barcode_reader.h"

namespace modeldeploy::vision::barcode {
    /*! @brief 条码/二维码识别器（纯 CV，零 DNN，跨全部后端） */
    class MODELDEPLOY_CXX_EXPORT BarcodeDetector {
    public:
        BarcodeDetector() = default;

        /*! 限定解码格式子集（FMT_* 的位或），默认 FMT_ALL。 */
        void set_formats(Formats formats);

        /*! 检测并解码图片中的所有码。返回可能为空。 */
        std::vector<BarcodeResult> detect(const ImageData& img) const;

        Formats formats() const { return formats_; }

    private:
        Formats formats_ = FMT_ALL;
    };
}

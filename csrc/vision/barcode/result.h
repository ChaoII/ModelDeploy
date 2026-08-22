#pragma once
#include <array>
#include <string>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::barcode {
    /*! @brief 条码 / 二维码解码结果 */
    struct MODELDEPLOY_CXX_EXPORT BarcodeResult {
        BarcodeResult() = default;
        std::string text;                    // 解码文本 / URL
        std::string format;                  // "QR_CODE"/"EAN_13"/"CODE_128"/"DATA_MATRIX"/...
        std::array<Point2f, 4> quad;         // 码区域四角定位框（左上起顺时针）
        float score = 0.0f;                  // 可靠度 [0,1]
        bool is_qr = false;                  // 是否二维码
    };
}

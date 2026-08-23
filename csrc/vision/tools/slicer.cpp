#include "vision/tools/slicer.h"
#include <algorithm>
namespace modeldeploy::vision::tool {
std::vector<Slice> InferenceSlicer::slice(const ImageData& img) const {
    const int W = img.width(), H = img.height();
    std::vector<Slice> out;
    if (W <= 0 || H <= 0) return out;
    const int step_w = std::max(1, tile_w_ - overlap_);
    const int step_h = std::max(1, tile_h_ - overlap_);
    for (int y = 0; y < H; y += step_h)
        for (int x = 0; x < W; x += step_w) {
            const int tw = std::min(tile_w_, W - x);
            const int th = std::min(tile_h_, H - y);
            const Rect2f box((float)x, (float)y, (float)tw, (float)th);
            out.push_back(Slice{img.crop(box), box});
        }
    return out;
}
void reassemble(const std::vector<Slice>& slices, const std::vector<Detections>& per_slice,
                ImageData* out, std::vector<Rect2f>* mapped_boxes) {
    if (slices.empty()) return;
    int W = 0, H = 0;
    for (const auto& s : slices) {
        W = std::max(W, (int)(s.offset.x + s.offset.width));
        H = std::max(H, (int)(s.offset.y + s.offset.height));
    }
    *out = ImageData(W, H, MdImageType::PKG_BGR_U8);
    mapped_boxes->clear();
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& dets = (i < per_slice.size()) ? per_slice[i] : Detections{};
        for (const auto& b : dets.boxes) {
            mapped_boxes->push_back(Rect2f(b.x + slices[i].offset.x, b.y + slices[i].offset.y, b.width, b.height));
        }
    }
}
} // namespace modeldeploy::vision::tool

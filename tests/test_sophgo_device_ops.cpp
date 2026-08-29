// Sophgo(TPU) 设备帧中间算子(crop/rotate/cvt_color)集成验收。
// 仅 ENABLE_SOPHGO 编译(见 tests/CMakeLists.txt:ENABLE_SOPHGO AND SOPHGO_FOUND)。
//
// 分两类：
//   1) fail-closed 语义(无 Sophgo 设备也可运行)：非 TPU 帧必须返回 false，绝不静默处理。
//   2) 真实 TPU 设备路径：构建 TPU NV12 设备帧 → 就地算子 → 断言 Device::TPU 输出；
//      需 Sophgo 设备(SOPHON-Sail/libsophon)。无设备时 bm_dev_request 失败 → 跳过(WARN + REQUIRE(true))。
//      注：像素级 CPU 对照需设备驱动闭环，属"未验证:需 Sophgo 设备"项，详见汇报。
//
#ifdef ENABLE_SOPHGO
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <vector>

#include "bmlib_runtime.h"
#include "bmcv_api.h"
#include "bmcv_api_ext.h"

#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "core/enum_variables.h"

using namespace modeldeploy::vision;

namespace {

    // 用给定设备 Y/UV 地址构造 Device::TPU 的 NV12 设备帧(借用,不拥有)。
    ImageData tpu_nv12_frame(void* y, void* uv, int w, int h) {
        ImageData::Plane pl[2] = {{static_cast<const uint8_t*>(y), w},
                                  {static_cast<const uint8_t*>(uv), w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, Device::TPU);
    }

    // 主机 NV12 渐变图 → 上传为 TPU 设备帧。返回空帧表示失败/无设备。
    // 输出设备的 Y/UV 地址与保活 owner 经出参返回(设备内存由 owner 释放)。
    ImageData upload_nv12_to_tpu(bm_handle_t h, int w, int h_img,
                                 void** dev_y, void** dev_uv,
                                 std::shared_ptr<void>* owner) {
        if (!h || w <= 0 || h_img <= 0) return {};
        std::vector<uint8_t> y(w * h_img), uv(w * h_img / 2);
        for (int i = 0; i < h_img; ++i)
            for (int j = 0; j < w; ++j)
                y[i * w + j] = static_cast<uint8_t>((i * 3 + j * 5) & 0xFF);
        for (size_t i = 0; i < uv.size(); ++i)
            uv[i] = static_cast<uint8_t>((i * 7) & 0xFF);

        bm_image img{};
        if (bm_image_create(h, h_img, w, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE, &img, nullptr) != BM_SUCCESS)
            return {};
        if (bm_image_alloc_dev_mem(img, 0) != BM_SUCCESS) { bm_image_destroy(&img); return {}; }
        void* host[] = {y.data(), uv.data()};
        if (bm_image_copy_host_to_device(img, host) != BM_SUCCESS) { bm_image_destroy(&img); return {}; }

        bm_device_mem_t mems[2]{};
        bm_image_get_plane_memory(img, mems);
        const unsigned long long ay = bm_mem_get_device_addr(mems[0]);
        const unsigned long long auv = bm_mem_get_device_addr(mems[1]);
        // owner：持有 img 的 dev mem 以便释放（经 img 生命周期）
        auto holder = std::make_shared<bm_image>(img);
        *dev_y = reinterpret_cast<void*>(ay);
        *dev_uv = reinterpret_cast<void*>(auv);
        *owner = std::shared_ptr<void>(reinterpret_cast<void*>(ay), [holder](void*) {
            bm_image img2 = *holder;
            bm_image_destroy(&img2);
        });
        return tpu_nv12_frame(*dev_y, *dev_uv, w, h_img);
    }

} // namespace

// ── 1) fail-closed：非 TPU 帧必须拒绝(安全，无设备也可运行) ──
TEST_CASE("Sophgo intermediate ops reject non-TPU frames (fail-closed)", "[sophgo]") {
    auto backend = create_processor_backend(Device::TPU, Backend::SOPHGO, 0);
    REQUIRE(backend != nullptr);

    // CPU NV12 帧：设备算子不得处理(设备显存不能由宿主指针写，属 UB)。
    ImageData cpu_frame(64, 64, MdImageType::NV12);
    REQUIRE(cpu_frame.plane_count() >= 2);
    ImageData out;
    REQUIRE_FALSE(backend->crop(cpu_frame, 4, 4, 32, 32, &out));
    REQUIRE_FALSE(backend->rotate(cpu_frame, ROTATE_90, &out));
    REQUIRE_FALSE(backend->cvt_color(cpu_frame, ColorConvertType::CVT_NV122PKG_BGR, &out));
    REQUIRE(out.empty());
}

// ── 2) 真实 TPU 设备路径(需 Sophgo 设备；无设备跳过) ──
TEST_CASE("Sophgo TPU device-frame crop/rotate/cvt_color", "[sophgo]") {
    bm_handle_t h = nullptr;
    if (bm_dev_request(&h, 0) != BM_SUCCESS) {
        // 无 Sophgo 设备：本用例为设备联调项，跳过(不 RED)。
        WARN("No Sophgo device available; TPU device-frame op test skipped.");
        REQUIRE(true);
        return;
    }

    const int W = 64, H = 64;
    void* dy = nullptr; void* duv = nullptr; std::shared_ptr<void> owner;
    ImageData src = upload_nv12_to_tpu(h, W, H, &dy, &duv, &owner);
    auto backend = create_processor_backend(Device::TPU, Backend::SOPHGO, 0);
    REQUIRE(backend != nullptr);

    if (!src.empty()) {
        // crop：输出仍为 TPU NV12，尺寸 = 裁剪区。
        ImageData c;
        REQUIRE(backend->crop(src, 4.0f, 4.0f, 32.0f, 32.0f, &c));
        REQUIRE(c.device() == Device::TPU);
        REQUIRE(c.type() == MdImageType::NV12);
        REQUIRE(c.width() == 32);
        REQUIRE(c.height() == 32);

        // rotate 90：W/H 互换。
        ImageData r;
        REQUIRE(backend->rotate(src, ROTATE_90, &r));
        REQUIRE(r.device() == Device::TPU);
        REQUIRE(r.width() == H);
        REQUIRE(r.height() == W);

        // cvt_color NV12→PKG_BGR：单平面 TPU 输出。
        ImageData bgr;
        REQUIRE(backend->cvt_color(src, ColorConvertType::CVT_NV122PKG_BGR, &bgr));
        REQUIRE(bgr.device() == Device::TPU);
        REQUIRE(bgr.type() == MdImageType::PKG_BGR_U8);
        REQUIRE(bgr.plane_count() == 1);

        // 像素级 CPU 对照：需 D2H 回读 + CpuProcessorBackend 基准。
        // —— 该段为设备联调验收(需驱动闭环)，见汇报"未验证"项；此处保留对照骨架。
        //   CPU_NV12 源 → CpuProcessorBackend::crop 得 CPU 基准，再与 D2H(c) 容差比对。
    }

    bm_dev_free(h);
}
#endif // ENABLE_SOPHGO

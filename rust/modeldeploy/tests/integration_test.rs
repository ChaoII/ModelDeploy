use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::types::*;

// ════════════════════════════════════════════════════════════════
// Image 函数全覆盖测试
// ════════════════════════════════════════════════════════════════

const TEST_IMG: &str = "../../test_data/test_images/test_detection0.jpg";

/// from_bgr24 + 属性
#[test]
fn test_image_from_bgr24() {
    let data = vec![0u8; 2 * 2 * 3];
    let img = Image::from_bgr24(&data, 2, 2);
    assert_eq!(img.width(), 2);
    assert_eq!(img.height(), 2);
    assert_eq!(img.channels(), 3);
    assert_eq!(img.data().len(), 12);
    println!("from_bgr24 OK");
}

/// read + save + clone + crop
#[test]
fn test_image_read_save_clone_crop() -> Result<()> {
    let img = Image::read(TEST_IMG)?;
    assert!(img.width() > 0);
    assert!(img.height() > 0);
    assert!(img.channels() == 3 || img.channels() == 1);

    let cloned = img.clone_image()?;
    assert_eq!(cloned.width(), img.width());
    assert_eq!(cloned.height(), img.height());

    // 裁剪左上 100x100
    let cropped = img.crop(0, 0, 100, 100)?;
    assert_eq!(cropped.width(), 100);
    assert_eq!(cropped.height(), 100);

    // 保存裁剪结果
    cropped.save("rust_test_crop.jpg")?;
    println!("read/save/clone/crop OK");
    Ok(())
}

/// from_rgb24
#[test]
fn test_image_from_rgb24() -> Result<()> {
    let rgb = vec![0u8; 4 * 4 * 3]; // 4x4 RGB
    let img = Image::from_rgb24(&rgb, 4, 4)?;
    assert_eq!(img.width(), 4);
    assert_eq!(img.height(), 4);
    assert_eq!(img.channels(), 3);
    println!("from_rgb24 OK");
    Ok(())
}

/// from_nv12
#[test]
fn test_image_from_nv12() -> Result<()> {
    let w = 4; let h = 4;
    let nv12 = vec![0u8; (w * h * 3 / 2) as usize];
    let img = Image::from_nv12(&nv12, w, h)?;
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    println!("from_nv12 OK");
    Ok(())
}

/// from_yuv420p
#[test]
fn test_image_from_yuv420p() -> Result<()> {
    let w = 4; let h = 4;
    let yuv = vec![0u8; (w * h * 3 / 2) as usize];
    let img = Image::from_yuv420p(&yuv, w, h)?;
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    println!("from_yuv420p OK");
    Ok(())
}

/// from_compressed / encode
#[test]
fn test_image_from_compressed() -> Result<()> {
    // 从文件读图，压缩后解码验证
    let img = Image::read(TEST_IMG)?;
    // 先验证 from_compressed: 从已存在的 jpg 读取
    let jpg_bytes = std::fs::read(TEST_IMG).unwrap_or_default();
    if !jpg_bytes.is_empty() {
        let decoded = Image::from_compressed(&jpg_bytes)?;
        assert!(decoded.width() > 0);
        assert!(decoded.height() > 0);
        println!("from_compressed OK ({} bytes, {}x{})", jpg_bytes.len(), decoded.width(), decoded.height());
    } else {
        // 没有测试图片时跳过
        println!("from_compressed SKIP (no test data)");
    }
    Ok(())
}

/// show（仅验证不崩溃 - 需要在有 GUI 的环境运行）
#[test]
#[ignore]
fn test_image_show() {
    let img = Image::from_bgr24(&[0u8; 3], 1, 1);
    img.show();
}

// ════════════════════════════════════════════════════════════════
// RuntimeOption 测试
// ════════════════════════════════════════════════════════════════

#[test]
fn test_runtime_option_default() {
    let opt = RuntimeOption::new();
    assert_eq!(opt.get_device_id(), 0);
}

#[test]
fn test_runtime_option_gpu() {
    let opt = RuntimeOption::new().gpu(0).fp16(true).ort_backend();
    assert_eq!(opt.get_backend_value(), 0);
    assert!(opt.is_fp16());
}

#[test]
fn test_runtime_option_trt() {
    let opt = RuntimeOption::new().gpu(1).trt_backend().fp16(true)
        .trt_min_shape("images:1x3x640x640").trt_cache("./cache");
    assert_eq!(opt.get_backend_value(), 2);
    assert_eq!(opt.get_device_id(), 1);
}

// ════════════════════════════════════════════════════════════════
// 类型测试
// ════════════════════════════════════════════════════════════════

#[test]
fn test_type_conversions() {
    let md_rect = modeldeploy::ffi::MDRect { x: 10, y: 20, width: 100, height: 200 };
    let rect: Rect = md_rect.into();
    assert_eq!(rect.x, 10);
    assert_eq!(rect.y, 20);
    let _back: modeldeploy::ffi::MDRect = rect.into();
}

#[test]
fn test_detection_result() {
    let det = Detection {
        rect: Rect { x: 1, y: 2, width: 100, height: 50 },
        label_id: 0, score: 0.95, label_name: "person".into(),
    };
    assert_eq!(det.score, 0.95);
}

#[test]
fn test_face_detection_result() {
    let face = FaceDetection {
        rect: Rect { x: 10, y: 10, width: 50, height: 50 },
        score: 0.98,
        landmarks: vec![Point3f { x: 20.0, y: 20.0, z: 0.0 }],
    };
    assert_eq!(face.landmarks.len(), 1);
}

// ════════════════════════════════════════════════════════════════
// 错误码测试
// ════════════════════════════════════════════════════════════════

#[test]
fn test_error_conversion() {
    use modeldeploy::ffi::*;
    use modeldeploy::error::*;

    let err = MdError::from(MDStatusCode_ModelInitializeFailed);
    assert!(matches!(err, MdError::ModelInitFailed(_)));

    let err = MdError::from(MDStatusCode_ModelPredictFailed);
    assert!(matches!(err, MdError::PredictFailed(_)));

    assert!(check_status(MDStatusCode_Success).is_ok());
    assert!(check_status(MDStatusCode_PathNotFound).is_err());
}

// ════════════════════════════════════════════════════════════════
// 内存泄漏压力测试
// ════════════════════════════════════════════════════════════════
// 反复分配/释放，如果内存持续增长说明有泄漏。
// 这些测试不是精确的泄漏检测（需要 valgrind/asan），
// 但可以作为 CI 上的快速检查。

/// Image 反复创建释放 1000 次
#[test]
fn test_image_repeated_alloc_free() {
    let data = vec![0u8; 640 * 480 * 3];
    for _ in 0..1000 {
        // from_bgr24（零拷贝，不分配）
        let img = Image::from_bgr24(&data, 640, 480);
        drop(img);
    }
    println!("Image repeated alloc/free 1000x OK");
}

/// Image read + drop 反复 100 次（每次分配 CAPI 内存）
#[test]
fn test_image_repeated_read_drop() -> Result<()> {
    for i in 0..100 {
        let img = Image::read("../../test_data/test_images/test_detection0.jpg")?;
        assert!(img.width() > 0);
        drop(img); // 显式释放 CAPI 内存
        if i == 0 { println!("First iteration OK"); }
    }
    println!("Image read/drop 100x OK");
    Ok(())
}

/// clone + crop 反复 100 次
#[test]
fn test_image_repeated_clone_crop() -> Result<()> {
    let img = Image::read("../../test_data/test_images/test_detection0.jpg")?;
    for _ in 0..100 {
        let cloned = img.clone_image()?;
        let cropped = cloned.crop(0, 0, 100, 100)?;
        assert_eq!(cropped.width(), 100);
        drop(cropped);
        drop(cloned);
    }
    println!("Image clone/crop 100x OK");
    Ok(())
}

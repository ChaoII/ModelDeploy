use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::types::*;

/// 测试图像创建（从内存数据，不需要文件）
#[test]
fn test_image_creation() -> Result<()> {
    // 创建一个 4x4 的 BGR 测试图像
    let data = vec![
        0u8, 0, 255,  // red pixel
        0, 255, 0,    // green pixel
        255, 0, 0,    // blue pixel
        128, 128, 128, // gray pixel
    ];
    let img = Image::from_bgr24(&data, 2, 2);
    assert_eq!(img.width(), 2);
    assert_eq!(img.height(), 2);
    assert_eq!(img.channels(), 3);
    assert_eq!(img.data().len(), 12);
    println!("图像创建测试通过: {}x{} {}ch", img.width(), img.height(), img.channels());
    Ok(())
}

/// 测试 RuntimeOption 默认值
#[test]
fn test_runtime_option_default() {
    let opt = RuntimeOption::new();
    // 默认设备 ID 为 0
    assert_eq!(opt.get_device_id(), 0);
}

/// 测试 RuntimeOption GPU 配置
#[test]
fn test_runtime_option_gpu() {
    let opt = RuntimeOption::new().gpu(0).fp16(true).ort_backend();
    assert_eq!(opt.get_backend_value(), 0); // MD_BACKEND_ORT
    assert_eq!(opt.is_fp16(), true);
}

/// 测试 RuntimeOption TRT 配置
#[test]
fn test_runtime_option_trt() {
    let opt = RuntimeOption::new()
        .gpu(1)
        .trt_backend()
        .fp16(true)
        .trt_min_shape("images:1x3x640x640")
        .trt_cache("./cache");
    assert_eq!(opt.get_backend_value(), 2); // MD_BACKEND_TRT
    assert_eq!(opt.get_device_id(), 1);
}

/// 测试类型转换
#[test]
fn test_type_conversions() {
    let md_rect = modeldeploy::ffi::MDRect {
        x: 10,
        y: 20,
        width: 100,
        height: 200,
    };
    let rect: Rect = md_rect.into();
    assert_eq!(rect.x, 10);
    assert_eq!(rect.y, 20);
    assert_eq!(rect.width, 100);
    assert_eq!(rect.height, 200);

    let back: modeldeploy::ffi::MDRect = rect.into();
    assert_eq!(back.x, 10);
}

/// 测试检测结果结构
#[test]
fn test_detection_result() {
    let det = Detection {
        rect: Rect { x: 1, y: 2, width: 100, height: 50 },
        label_id: 0,
        score: 0.95,
        label_name: "person".into(),
    };
    assert_eq!(det.score, 0.95);
    assert_eq!(det.label_id, 0);
    assert_eq!(det.rect.width, 100);
}

/// 测试人脸检测结果
#[test]
fn test_face_detection_result() {
    let face = FaceDetection {
        rect: Rect { x: 10, y: 10, width: 50, height: 50 },
        score: 0.98,
        landmarks: vec![
            Point3f { x: 20.0, y: 20.0, z: 0.0 },
            Point3f { x: 40.0, y: 20.0, z: 0.0 },
        ],
    };
    assert_eq!(face.landmarks.len(), 2);
    assert_eq!(face.score, 0.98);
}

/// 测试错误码转换
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

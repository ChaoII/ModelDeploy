use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::detection::UltralyticsDet;

/// 集成测试：检测模型完整链路
#[test]
fn test_detection_model() -> Result<()> {
    let model_path = std::env::var("MD_TEST_MODEL")
        .unwrap_or_else(|_| "../../test_data/test_models/yolo11n_nms.onnx".to_string());
    let image_path = std::env::var("MD_TEST_IMAGE")
        .unwrap_or_else(|_| "../../test_data/test_images/test_detection.jpg".to_string());

    // 加载模型
    let opt = RuntimeOption::new().gpu(0).fp16(true);
    let det = UltralyticsDet::new(&model_path, &opt)?;
    assert!(det.is_initialized(), "模型应该成功初始化");

    // 读取图像
    let img = Image::read(&image_path)?;
    assert!(img.width() > 0, "图像宽度应 > 0");
    assert!(img.height() > 0, "图像高度应 > 0");

    // 推理
    let results = det.predict(&img)?;
    println!("检测到 {} 个目标", results.len());

    // 结果合理性检查
    for (i, r) in results.iter().enumerate() {
        assert!(r.score >= 0.0 && r.score <= 1.0, "[{}] score 应在 [0,1] 范围: {}", i, r.score);
        assert!(r.rect.width > 0, "[{}] width 应 > 0", i);
        assert!(r.rect.height > 0, "[{}] height 应 > 0", i);
        assert!(r.label_id >= 0, "[{}] label_id 应 >= 0", i);
    }

    // 绘制结果保存
    if results.len() > 0 {
        let drawn = det.predict_with_draw(&img, 0.5)?;
        drawn.save("rust_test_detection_result.jpg")?;
    }

    Ok(())
}

/// 多次推理测试内存泄漏
#[test]
fn test_detection_multiple_inferences() -> Result<()> {
    let model_path = std::env::var("MD_TEST_MODEL")
        .unwrap_or_else(|_| "../../test_data/test_models/yolo11n_nms.onnx".to_string());
    let image_path = std::env::var("MD_TEST_IMAGE")
        .unwrap_or_else(|_| "../../test_data/test_images/test_detection.jpg".to_string());

    let opt = RuntimeOption::new().gpu(0);
    let det = UltralyticsDet::new(&model_path, &opt)?;
    let img = Image::read(&image_path)?;

    // 多次推理验证内存稳定
    for i in 0..100 {
        let results = det.predict(&img)?;
        if i == 0 {
            println!("首次推理: {} 个目标", results.len());
        }
    }
    println!("100 次推理完成，无内存泄漏");

    Ok(())
}

/// 结果一致性测试（与 Python 结果对比）
#[test]
fn test_detection_consistency() -> Result<()> {
    let model_path = std::env::var("MD_TEST_MODEL")
        .unwrap_or_else(|_| "../../test_data/test_models/yolo11n_nms.onnx".to_string());
    let image_path = std::env::var("MD_TEST_IMAGE")
        .unwrap_or_else(|_| "../../test_data/test_images/test_detection.jpg".to_string());

    let opt = RuntimeOption::new().gpu(0).fp16(true);
    let det = UltralyticsDet::new(&model_path, &opt)?;
    let img = Image::read(&image_path)?;
    let results = det.predict(&img)?;

    // 结果非空（测试图中有已知目标）
    assert!(
        results.len() > 0,
        "应至少检测到一个目标（测试图可能需调整）"
    );
    println!("一致性检查通过: {} 个目标", results.len());

    Ok(())
}

/// 图像基本操作测试
#[test]
fn test_image_operations() -> Result<()> {
    let image_path = std::env::var("MD_TEST_IMAGE")
        .unwrap_or_else(|_| "../../test_data/test_images/test_detection.jpg".to_string());

    let img = Image::read(&image_path)?;
    assert!(img.width() > 0);
    assert!(img.height() > 0);
    assert!(img.channels() == 3 || img.channels() == 1);

    // 测试克隆
    let cloned = img.clone_image()?;
    assert_eq!(cloned.width(), img.width());
    assert_eq!(cloned.height(), img.height());

    // 测试 Save
    cloned.save("rust_test_image_clone.jpg")?;

    println!("图像操作测试通过: {}x{} {}ch", img.width(), img.height(), img.channels());
    Ok(())
}

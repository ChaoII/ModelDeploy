use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::detection::UltralyticsDet;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "../../test_data/test_models/yolo11n_nms.onnx".to_string()
    });
    let image_path = std::env::args().nth(2).unwrap_or_else(|| {
        "../../test_data/test_images/test_detection0.jpg".to_string()
    });

    if !Path::new(&model_path).exists() {
        eprintln!("模型文件不存在: {}", model_path);
        return Ok(());
    }
    if !Path::new(&image_path).exists() {
        eprintln!("图片文件不存在: {}", image_path);
        return Ok(());
    }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsDet::new(&model_path, &opt)?;
    println!("模型加载成功: {}", model_path);

    let img = Image::read(&image_path)?;
    println!("图像: {}x{} {}通道", img.width(), img.height(), img.channels());

    let results = model.predict(&img)?;
    println!("检测到 {} 个目标:", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  [{}/{}] label={} score={:.4} rect=[{}x{} {}x{}]",
                 i+1, results.len(), r.label_id, r.score,
                 r.rect.x, r.rect.y, r.rect.width, r.rect.height);
    }
    Ok(())
}

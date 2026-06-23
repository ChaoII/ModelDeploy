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
    if !Path::new(&model_path).exists() { eprintln!("模型不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片不存在: {}", image_path); return Ok(()); }

    // GPU + enable_trt = false
    let opt = RuntimeOption::new()
        .gpu(0)
        .fp16(true)
        .enable_trt(false)
        .ort_backend();

    println!("enable_trt = false");
    let model = UltralyticsDet::new(&model_path, &opt)?;
    println!("模型加载成功");
    let img = Image::read(&image_path)?;
    let results = model.predict(&img)?;
    println!("检测到 {} 个目标", results.len());
    println!("检测到 {:?} 个目标", results);
    Ok(())
}

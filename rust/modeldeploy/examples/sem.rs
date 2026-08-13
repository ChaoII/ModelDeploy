use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::sem::UltralyticsSem;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/onnx/yolo26n/yolo26n-sem.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/test_sem.jpg".into());
    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsSem::new(&model_path, &opt)?;
    println!("语义分割模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let result = model.predict(&img)?;
    let dims = if result.shape.len() >= 2 { format!("{}x{}", result.shape[0], result.shape[1]) } else { "?".into() };
    println!("语义分割完成: 尺寸={}, 类别数={}, 像素数={}", dims, result.num_classes, result.labels.len());
    Ok(())
}

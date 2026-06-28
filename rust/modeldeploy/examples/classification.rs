use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::classification::UltralyticsCls;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/yolo11n_cls.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/111.jpg".into());
    let topk: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsCls::new(&model_path, &opt)?;
    println!("分类模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let results = model.predict(&img, topk)?;
    println!("Top-{} 分类结果:", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  [{}/{}] label={} score={:.4}", i+1, results.len(), r.label_id, r.score);
    }
    Ok(())
}

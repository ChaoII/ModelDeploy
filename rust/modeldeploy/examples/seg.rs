use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::obb_iseg_pose::UltralyticsSeg;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/yolo11n_seg.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/test_seg.jpg".into());
    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsSeg::new(&model_path, &opt)?;
    println!("分割模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let results = model.predict(&img)?;
    println!("分割到 {} 个目标", results.len());
    for (i, r) in results.iter().enumerate() {
        let dims = if r.mask_shape.len() >= 2 { format!("{}x{}", r.mask_shape[0], r.mask_shape[1]) } else { "?".into() };
        println!("  [{}/{}] label={} score={:.4} mask={}", i+1, results.len(), r.label_id, r.score, dims);
    }
    Ok(())
}

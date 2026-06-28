use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face::Scrfd;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/scrfd_10g.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/test_face_det.jpg".into());
    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = Scrfd::new(&model_path, &opt)?;
    println!("人脸检测模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let faces = model.predict(&img)?;
    println!("检测到 {} 张人脸", faces.len());
    for (i, f) in faces.iter().enumerate() {
        print!("  [{}/{}] score={:.4} rect=[{}x{} {}x{}]", i+1, faces.len(), f.score,
               f.rect.x, f.rect.y, f.rect.width, f.rect.height);
        if !f.landmarks.is_empty() { print!(" landmarks={}", f.landmarks.len()); }
        println!();
    }
    Ok(())
}

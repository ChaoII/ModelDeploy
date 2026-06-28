use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face_extra::FaceAge;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/face_age.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/test_face_age.jpg".into());
    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = FaceAge::new(&model_path, &opt)?;
    println!("年龄模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let age = model.predict(&img)?;
    println!("预测年龄: {}", age);
    Ok(())
}

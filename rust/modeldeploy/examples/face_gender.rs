use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face_extra::FaceGender;
use std::path::Path;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../../test_data/test_models/face_gender.onnx".into());
    let image_path = std::env::args().nth(2).unwrap_or_else(|| "../../test_data/test_images/test_face_gender.jpg".into());
    if !Path::new(&model_path).exists() { eprintln!("模型文件不存在: {}", model_path); return Ok(()); }
    if !Path::new(&image_path).exists() { eprintln!("图片文件不存在: {}", image_path); return Ok(()); }

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = FaceGender::new(&model_path, &opt)?;
    println!("性别模型加载成功");
    let img = Image::read(&image_path)?;
    println!("图像: {}x{}", img.width(), img.height());
    let gender = model.predict(&img)?;
    println!("预测性别: {:?}", gender);
    Ok(())
}

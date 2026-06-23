use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face_extra::FaceGender;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = FaceGender::new(&args[1], &opt)?;
    println!("性别模型加载成功");
    let img = Image::read(&args[2])?;
    println!("图像: {}x{}", img.width(), img.height());
    let gender = model.predict(&img)?;
    println!("预测性别: {:?}", gender);
    Ok(())
}

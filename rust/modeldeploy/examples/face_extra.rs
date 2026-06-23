use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face_extra::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let example = std::env::current_exe()?.file_name().unwrap().to_string_lossy().to_string();

    if example.contains("age") {
        if args.len() < 3 { eprintln!("用法: face_age <模型> <图片>"); return Ok(()); }
        let model = FaceAge::new(&args[1], &RuntimeOption::new().gpu(0).ort_backend())?;
        let img = Image::read(&args[2])?;
        let age = model.predict(&img)?;
        println!("预测年龄: {}", age);
    } else if example.contains("gender") {
        if args.len() < 3 { eprintln!("用法: face_gender <模型> <图片>"); return Ok(()); }
        let model = FaceGender::new(&args[1], &RuntimeOption::new().gpu(0).ort_backend())?;
        let img = Image::read(&args[2])?;
        let gender = model.predict(&img)?;
        println!("预测性别: {:?}", gender);
    } else if example.contains("facerec") {
        if args.len() < 3 { eprintln!("用法: face_rec <模型> <图片>"); return Ok(()); }
        let model = FaceRec::new(&args[1], &RuntimeOption::new().gpu(0).ort_backend())?;
        let img = Image::read(&args[2])?;
        let result = model.predict(&img)?;
        println!("特征维度: {}", result.embedding.len());
    }
    Ok(())
}

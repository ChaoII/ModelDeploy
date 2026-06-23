use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face::Scrfd;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = Scrfd::new(&args[1], &opt)?;
    println!("人脸检测模型加载成功");
    let img = Image::read(&args[2])?;
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

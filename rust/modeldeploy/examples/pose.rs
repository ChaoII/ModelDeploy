use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::obb_iseg_pose::UltralyticsPose;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsPose::new(&args[1], &opt)?;
    println!("姿态模型加载成功");
    let img = Image::read(&args[2])?;
    println!("图像: {}x{}", img.width(), img.height());
    let results = model.predict(&img)?;
    println!("检测到 {} 个人", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  [{}/{}] score={:.4} keypoints={}", i+1, results.len(), r.score, r.keypoints.len());
    }
    Ok(())
}

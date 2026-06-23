use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::obb_iseg_pose::UltralyticsObb;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsObb::new(&args[1], &opt)?;
    println!("OBB 模型加载成功");
    let img = Image::read(&args[2])?;
    println!("图像: {}x{}", img.width(), img.height());
    let results = model.predict(&img)?;
    println!("检测到 {} 个目标", results.len());
    for r in &results {
        println!("  label={} score={:.4} angle={:.2} center=({:.0},{:.0})", r.label_id, r.score, r.angle, r.xc, r.yc);
    }
    Ok(())
}

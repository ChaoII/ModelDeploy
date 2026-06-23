use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::obb_iseg_pose::UltralyticsSeg;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsSeg::new(&args[1], &opt)?;
    println!("分割模型加载成功");
    let img = Image::read(&args[2])?;
    println!("图像: {}x{}", img.width(), img.height());
    let results = model.predict(&img)?;
    println!("分割到 {} 个目标", results.len());
    for (i, r) in results.iter().enumerate() {
        let mask_dims = if r.mask_shape.len() >= 2 { format!("{}x{}", r.mask_shape[0], r.mask_shape[1]) } else { "?".into() };
        println!("  [{}/{}] label={} score={:.4} mask={}", i+1, results.len(), r.label_id, r.score, mask_dims);
    }
    Ok(())
}

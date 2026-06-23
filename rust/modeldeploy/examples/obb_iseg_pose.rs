use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::obb_iseg_pose::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("用法: obb <模型路径> <图片路径>");
    let image_path = args.get(2).expect("用法: obb <模型路径> <图片路径>");
    let example = std::env::current_exe()?.file_name().unwrap().to_string_lossy().to_string();

    let opt = RuntimeOption::new().gpu(0).ort_backend();

    if example.contains("obb") {
        let model = UltralyticsObb::new(model_path, &opt)?;
        let img = Image::read(image_path)?;
        let results = model.predict(&img)?;
        println!("OBB 检测到 {} 个目标", results.len());
        for r in &results { println!("  label={} score={:.4} angle={:.2}", r.label_id, r.score, r.angle); }
    } else if example.contains("seg") {
        let model = UltralyticsSeg::new(model_path, &opt)?;
        let img = Image::read(image_path)?;
        let results = model.predict(&img)?;
        println!("分割检测到 {} 个目标", results.len());
        for r in &results { println!("  label={} score={:.4} mask={}x{}", r.label_id, r.score,
            r.mask_shape.get(0).unwrap_or(&0), r.mask_shape.get(1).unwrap_or(&0)); }
    } else if example.contains("pose") {
        let model = UltralyticsPose::new(model_path, &opt)?;
        let img = Image::read(image_path)?;
        let results = model.predict(&img)?;
        println!("姿态检测到 {} 个人", results.len());
        for r in &results { println!("  score={:.4} keypoints={}", r.score, r.keypoints.len()); }
    }
    Ok(())
}

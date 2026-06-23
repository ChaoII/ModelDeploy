use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::classification::UltralyticsCls;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: {} <模型路径> <图片路径> [topk]", args[0]);
        return Ok(());
    }
    let model_path = &args[1];
    let image_path = &args[2];
    let topk: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = UltralyticsCls::new(model_path, &opt)?;
    println!("分类模型加载成功");

    let img = Image::read(image_path)?;
    println!("图像: {}x{}", img.width(), img.height());

    let results = model.predict(&img, topk)?;
    println!("Top-{} 分类结果:", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  [{}/{}] label={} score={:.4}", i+1, results.len(), r.label_id, r.score);
    }
    Ok(())
}

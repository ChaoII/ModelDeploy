use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::detection::UltralyticsDet;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "yolo11n.onnx".to_string());
    let image_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "test.jpg".to_string());

    println!("加载模型: {}", model_path);
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let det = UltralyticsDet::new(&model_path, &opt)?;
    println!("模型加载成功");

    println!("读取图像: {}", image_path);
    let img = Image::read(&image_path)?;
    println!("图像尺寸: {}x{} {}通道", img.width(), img.height(), img.channels());

    println!("推理中...");
    let results = det.predict(&img)?;
    println!("检测到 {} 个目标", results.len());

    for (i, det) in results.iter().enumerate() {
        println!(
            "  [{}/{}] label={} score={:.4} rect=[{}x{} {}x{}]",
            i + 1,
            results.len(),
            det.label_id,
            det.score,
            det.rect.x, det.rect.y, det.rect.width, det.rect.height,
        );
    }

    println!("推理完成");
    Ok(())
}

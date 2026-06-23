use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::detection::UltralyticsDet;

fn main() -> Result<()> {
    // 1. 加载模型
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

    // 2. 读取图像
    println!("读取图像: {}", image_path);
    let img = Image::read(&image_path)?;
    println!(
        "图像尺寸: {}x{} {}通道",
        img.width(),
        img.height(),
        img.channels()
    );

    // 3. 推理
    println!("推理中...");
    let results = det.predict(&img)?;
    println!("检测到 {} 个目标", results.len());

    // 4. 输出结果
    for (i, det) in results.iter().enumerate() {
        println!(
            "  [{}/{}] label={} score={:.4} rect=[{}x{} {}x{}]",
            i + 1,
            results.len(),
            det.label_id,
            det.score,
            det.rect.x,
            det.rect.y,
            det.rect.width,
            det.rect.height,
        );
    }

    // 5. 带绘制保存
    let output_path = "result.jpg";
    let drawn = det.predict_with_draw(&img, 0.5)?;
    drawn.save(output_path)?;
    println!("结果保存到: {}", output_path);

    Ok(())
}

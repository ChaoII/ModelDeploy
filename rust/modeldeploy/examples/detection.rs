use anyhow::Result;
use modeldeploy::{DrawOptions, Image, RuntimeOption, UltralyticsDet};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?.set_cpu_threads(4)?;

    let model = UltralyticsDet::new(&test_data("test_models/onnx/yolo11n/yolo11n.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_detection0.jpg"))?;

    // 纯推理
    let dets = model.predict(&img)?;
    println!("detected {} objects", dets.len());
    for d in &dets {
        println!("  [{:?}] score={:.3}", d.rect, d.score);
    }

    // 需要可视化时：句柄直达 C++ vis_det
    let canvas = img.clone()?;
    model.predict_and_draw(
        &img,
        &canvas,
        &DrawOptions::new()
            .with_threshold(0.4)
            .with_label_map(vec![
                (0, "person".into()),
                (1, "bicycle".into()),
                (2, "car".into()),
            ])
            .with_alpha(0.3),
    )?;
    canvas.save("detection_annotated.png")?;
    println!("visualized -> detection_annotated.png");

    Ok(())
}

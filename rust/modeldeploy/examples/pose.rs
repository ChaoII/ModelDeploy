use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, UltralyticsPose};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = UltralyticsPose::new(&test_data("test_models/onnx/yolo11n/yolo11n-pose.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/bus.jpg"))?;
    for p in model.predict(&img)? {
        println!("person: score={:.3} keypoints={}", p.score, p.keypoints.len());
    }
    Ok(())
}

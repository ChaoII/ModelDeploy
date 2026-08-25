use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, UltralyticsSeg};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = UltralyticsSeg::new(&test_data("test_models/onnx/yolo11n/yolo11n-seg.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_detection0.jpg"))?;
    println!("instances: {}", model.predict(&img)?.len());
    Ok(())
}

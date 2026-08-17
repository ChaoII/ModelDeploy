use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, UltralyticsObb};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU);
    let model = UltralyticsObb::new(&test_data("test_models/onnx/yolo11n/yolo11n-obb.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/bus.jpg"))?;
    for o in model.predict(&img)? {
        println!("obb: label={} score={:.3}", o.label_id, o.score);
    }
    Ok(())
}

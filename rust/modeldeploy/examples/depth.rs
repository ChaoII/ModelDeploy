use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, UltralyticsDepth};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = UltralyticsDepth::new(&test_data("test_models/onnx/yolo26n/yolo26n-depth.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/bus.jpg"))?;
    let depth = model.predict(&img)?;
    println!("depth: {}x{}", depth[0].width, depth[0].height);
    Ok(())
}

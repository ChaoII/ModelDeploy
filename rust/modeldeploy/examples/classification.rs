use anyhow::Result;
use modeldeploy::{Classification, Image, RuntimeOption};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = Classification::new(&test_data("test_models/onnx/yolo11n/yolo11n-cls.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/bus.jpg"))?;
    for c in model.predict(&img)? {
        println!("label={} score={:.3}", c.label_id, c.score);
    }
    Ok(())
}

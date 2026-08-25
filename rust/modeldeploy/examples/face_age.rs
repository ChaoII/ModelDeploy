use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, SeetaFaceAge};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = SeetaFaceAge::new(&test_data("test_models/onnx/face/age_predictor.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_face_id1.jpg"))?;
    println!("age: {}", model.predict(&img)?);
    Ok(())
}

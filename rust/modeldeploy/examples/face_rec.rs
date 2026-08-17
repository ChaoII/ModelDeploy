use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, SeetaFaceID};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU);
    let model = SeetaFaceID::new(&test_data("test_models/onnx/face/face_recognizer.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_face_id.jpg"))?;
    println!("embedding dim: {}", model.predict(&img)?[0].embedding.len());
    Ok(())
}

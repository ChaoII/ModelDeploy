use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, SeetaFaceGender};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = SeetaFaceGender::new(&test_data("test_models/onnx/seetaface/gender_predictor.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_face_gender.jpg"))?;
    println!("gender: {}", model.predict(&img)?);
    Ok(())
}

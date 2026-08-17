use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, UltralyticsSem};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU);
    let model = UltralyticsSem::new(&test_data("test_models/onnx/yolo26n/yolo26n-sem.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/bus.jpg"))?;
    let sem = model.predict(&img)?;
    println!("sem seg: {}x{} classes={}", sem[0].width, sem[0].height, sem[0].num_classes);
    Ok(())
}

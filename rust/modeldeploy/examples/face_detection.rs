use anyhow::Result;
use modeldeploy::{Image, RuntimeOption, Scrfd};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn main() -> Result<()> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU, 0)?;
    let model = Scrfd::new(&test_data("test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx"), &opt)?;
    let img = Image::read(&test_data("test_images/test_face_detection4.jpg"))?;
    for f in model.predict(&img)? {
        println!("face: score={:.3} landmarks={}", f.score, f.keypoints.len());
    }
    Ok(())
}

use anyhow::Result;
use modeldeploy::image::Image;
use modeldeploy::runtime::RuntimeOption;
use modeldeploy::vision::face_extra::FaceRec;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 { eprintln!("用法: {} <模型路径> <图片路径>", args[0]); return Ok(()); }
    let opt = RuntimeOption::new().gpu(0).ort_backend();
    let model = FaceRec::new(&args[1], &opt)?;
    println!("人脸识别模型加载成功");
    let img = Image::read(&args[2])?;
    println!("图像: {}x{}", img.width(), img.height());
    let result = model.predict(&img)?;
    println!("特征维度: {} (embedding)", result.embedding.len());
    if result.embedding.len() > 0 {
        let norm: f32 = result.embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
        println!("特征模长: {:.4}", norm);
    }
    Ok(())
}

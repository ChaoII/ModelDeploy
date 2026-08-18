use anyhow::Result;
use modeldeploy::{
    Classification, DbDetectorModel, DrawOptions, FaceRecognizerPipelineModel,
    Image, InsightFaceAnalysis, InsightFaceDetModel, Kokoro, LprDetectionModel, LprPipeline,
    LprRecognizerModel, PaddleOCR, PedestrianAttribute, RecognizerModel,
    RuntimeOption, Scrfd, SeetaFaceAge, SeetaFaceGender, SeetaFaceID, SenseVoice, UltralyticsDepth,
    UltralyticsDet, UltralyticsObb, UltralyticsPose, UltralyticsSeg, UltralyticsSem,
};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn test_img(rel: &str) -> String {
    format!("{}/../../test_data/test_images/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn cpu_opt() -> Result<RuntimeOption> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort().set_device(modeldeploy::ffi::MDDevice::CPU).set_cpu_threads(4);
    Ok(opt)
}

// ═══ Image ═══

#[test]
fn test_image_basics() -> Result<()> {
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    assert!(img.width() > 0 && img.height() > 0);

    let clone = img.clone()?;
    assert_eq!(clone.width(), img.width());
    assert_eq!(clone.height(), img.height());

    let crop = img.crop(10, 10, 50, 50)?;
    assert_eq!(crop.width(), 50);
    assert_eq!(crop.height(), 50);

    let jpg = img.encode(".jpg")?;
    assert!(!jpg.is_empty());
    assert_eq!(jpg[0], 0xFF);
    assert_eq!(jpg[1], 0xD8);

    let tmp = std::env::temp_dir().join("md_rust_test.png");
    let tmp = tmp.to_string_lossy().to_string();
    img.save(&tmp)?;
    std::fs::remove_file(&tmp).ok();
    Ok(())
}

#[test]
fn test_image_from_bgr24() -> Result<()> {
    let data = vec![0u8; 20 * 10 * 3];
    let img = Image::from_bgr24(&data, 20, 10)?;
    assert_eq!(img.width(), 20);
    assert_eq!(img.height(), 10);
    Ok(())
}

// ═══ Detection ═══

#[test]
fn test_detection() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsDet::new(&test_data("test_models/onnx/yolo11n/yolo11n.onnx"), &opt)?;
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let dets = model.predict(&img)?;
    assert!(!dets.is_empty(), "should detect objects");
    assert!(dets[0].score > 0.0 && dets[0].score <= 1.0);
    assert!(dets[0].rect.width > 0.0);
    Ok(())
}

#[test]
fn test_detection_clone() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsDet::new(&test_data("test_models/onnx/yolo11n/yolo11n.onnx"), &opt)?;
    let cloned = model.clone()?;
    assert!(cloned.is_ready());
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let a = model.predict(&img)?;
    let b = cloned.predict(&img)?;
    assert_eq!(a.len(), b.len());
    assert!((a[0].score - b[0].score).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_detection_draw() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsDet::new(&test_data("test_models/onnx/yolo11n/yolo11n.onnx"), &opt)?;
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let canvas = img.clone()?;
    let options = DrawOptions::new()
        .with_threshold(0.4)
        .with_label_map(vec![(0, "person".into()), (1, "bicycle".into()), (2, "car".into())])
        .with_alpha(0.3);
    let dets = model.predict_and_draw(&img, &canvas, &options)?;
    assert!(!dets.is_empty());
    let tmp = std::env::temp_dir().join("md_rust_draw.png");
    let tmp = tmp.to_string_lossy().to_string();
    canvas.save(&tmp)?;
    std::fs::remove_file(&tmp).ok();
    Ok(())
}

// ═══ Classification ═══

#[test]
fn test_classification() -> Result<()> {
    let opt = cpu_opt()?;
    let model = Classification::new(&test_data("test_models/onnx/yolo11n/yolo11n-cls.onnx"), &opt)?;
    let img = Image::read(&test_img("bus.jpg"))?;
    let cls = model.predict(&img)?;
    assert!(!cls.is_empty());
    assert!(cls[0].label_id >= 0);
    Ok(())
}

// ═══ Pose ═══

#[test]
fn test_pose() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsPose::new(&test_data("test_models/onnx/yolo11n/yolo11n-pose.onnx"), &opt)?;
    let img = Image::read(&test_img("bus.jpg"))?;
    let poses = model.predict(&img)?;
    assert!(!poses.is_empty());
    assert!(poses[0].keypoints.len() > 0);
    Ok(())
}

// ═══ OBB ═══

#[test]
fn test_obb() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsObb::new(&test_data("test_models/onnx/yolo11n/yolo11n-obb.onnx"), &opt)?;
    let img = Image::read(&test_img("bus.jpg"))?;
    let obbs = model.predict(&img)?;
    let _ = obbs;
    Ok(())
}

// ═══ InstanceSeg ═══

#[test]
fn test_instance_seg() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsSeg::new(&test_data("test_models/onnx/yolo11n/yolo11n-seg.onnx"), &opt)?;
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let segs = model.predict(&img)?;
    assert!(!segs.is_empty());
    Ok(())
}

// ═══ SemSeg ═══

#[test]
fn test_sem_seg() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsSem::new(&test_data("test_models/onnx/yolo26n/yolo26n-sem.onnx"), &opt)?;
    let img = Image::read(&test_img("bus.jpg"))?;
    let sem = model.predict(&img)?;
    assert!(!sem.is_empty());
    assert!(sem[0].labels.len() > 0);
    assert!(sem[0].num_classes > 0);
    Ok(())
}

// ═══ Depth ═══

#[test]
fn test_depth() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsDepth::new(&test_data("test_models/onnx/yolo26n/yolo26n-depth.onnx"), &opt)?;
    let img = Image::read(&test_img("bus.jpg"))?;
    let depth = model.predict(&img)?;
    assert!(depth[0].depth.len() > 0);
    Ok(())
}

// ═══ FaceDet ═══

#[test]
fn test_face_detection() -> Result<()> {
    let opt = cpu_opt()?;
    let model = Scrfd::new(&test_data("test_models/onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_detection4.jpg"))?;
    let faces = model.predict(&img)?;
    assert!(!faces.is_empty());
    assert!(faces[0].keypoints.len() > 0);
    Ok(())
}

// ═══ FaceRec ═══

#[test]
fn test_face_recognition() -> Result<()> {
    let opt = cpu_opt()?;
    let model = SeetaFaceID::new(&test_data("test_models/onnx/face/face_recognizer.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_id.jpg"))?;
    let rec = model.predict(&img)?;
    assert!(rec[0].embedding.len() > 0);
    Ok(())
}

// ═══ FaceAge / Gender ═══

#[test]
fn test_face_age_gender() -> Result<()> {
    let opt = cpu_opt()?;
    let age_model = SeetaFaceAge::new(&test_data("test_models/onnx/face/age_predictor.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_id1.jpg"))?;
    let age = age_model.predict(&img)?;
    assert!(age[0] >= 0);

    let gender_model = SeetaFaceGender::new(&test_data("test_models/onnx/face/gender_predictor.onnx"), &opt)?;
    let img2 = Image::read(&test_img("test_face_gender.jpg"))?;
    let gender = gender_model.predict(&img2)?;
    assert!(gender[0] >= 0);
    Ok(())
}

// ═══ InsightFace ═══

#[test]
fn test_insightface() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/insightface/buffalo_l");
    let path = format!("{}/det_10g.onnx|{}/w600k_r50.onnx|{}/2d106det.onnx|{}/1k3d68.onnx|{}/genderage.onnx",
        dir, dir, dir, dir, dir);
    let model = InsightFaceAnalysis::new(&path, &opt)?;
    let img = Image::read(&test_img("test_face1.jpg"))?;
    let faces = model.predict(&img)?;
    assert!(!faces.is_empty());
    assert!(faces[0].embedding.len() > 0);
    assert!(faces[0].gender >= 0);
    Ok(())
}

// ═══ OCR ═══

#[test]
fn test_ocr() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/ocr/ppocrv4_mobile");
    let dict = test_data("ppocrv4_dict.txt");
    let path = format!("{}/det_infer.onnx|{}/cls_infer.onnx|{}/rec_infer.onnx|{}", dir, dir, dir, dict);
    let model = PaddleOCR::new(&path, &opt)?;
    let img = Image::read(&test_img("test_ocr.png"))?;
    let lines = model.predict(&img)?;
    assert!(!lines.is_empty());
    assert!(!lines[0].text.is_empty());
    Ok(())
}

// ═══ LPR ═══

#[test]
fn test_lpr() -> Result<()> {
    let opt = cpu_opt()?;
    let det = test_data("test_models/onnx/yolov5plate.onnx");
    let rec = test_data("test_models/onnx/plate_recognition_color.onnx");
    let path = format!("{}|{}", det, rec);
    let model = LprPipeline::new(&path, &opt)?;
    let img = Image::read(&test_img("test_lpr_pipeline2.jpg"))?;
    let plates = model.predict(&img)?;
    let _ = plates;
    Ok(())
}

// ═══ PedestrianAttribute ═══

#[test]
fn test_pedestrian_attribute() -> Result<()> {
    let opt = cpu_opt()?;
    let det = test_data("test_models/onnx/zhgd_det.onnx");
    let cls = test_data("test_models/onnx/zhgd_ml.onnx");
    let path = format!("{}|{}", det, cls);
    let model = PedestrianAttribute::new(&path, &opt)?;
    model.set_input_size(1280, 1280)?;
    model.set_cls_input_size(192, 256)?;
    let img = Image::read(&test_img("test_pedestrian_attribute1.jpg"))?;
    let attrs = model.predict(&img)?;
    assert!(!attrs.is_empty());
    Ok(())
}

// ═══ 子模型测试 ═══

#[test]
fn test_ocr_det_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let model = DbDetectorModel::new(
        &test_data("test_models/onnx/ocr/ppocrv4_mobile/det_infer.onnx"),
        &opt,
    )?;
    let img = Image::read(&test_img("test_ocr.png"))?;
    let boxes = model.predict(&img)?;
    assert!(!boxes.is_empty());
    Ok(())
}

#[test]
fn test_ocr_rec_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/ocr/ppocrv4_mobile");
    let dict = test_data("ppocrv4_dict.txt");
    let path = format!("{}/rec_infer.onnx|{}", dir, dict);
    let model = RecognizerModel::new(&path, &opt)?;
    let img = Image::read(&test_img("test_ocr_recognition1.jpg"))?;
    let lines = model.predict(&img)?;
    let _ = lines;
    Ok(())
}

#[test]
fn test_face_rec_pipeline_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/face");
    let path = format!("{}/scrfd_2.5g_bnkps_shape640x640.onnx|{}/face_recognizer.onnx", dir, dir);
    let model = FaceRecognizerPipelineModel::new(&path, &opt)?;
    let img = Image::read(&test_img("test_face_detection4.jpg"))?;
    let recs = model.predict(&img)?;
    assert!(!recs.is_empty());
    assert!(recs[0].embedding.len() > 0);
    Ok(())
}

#[test]
fn test_insightface_det_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let model = InsightFaceDetModel::new(
        &test_data("test_models/onnx/insightface/buffalo_l/det_10g.onnx"),
        &opt,
    )?;
    let img = Image::read(&test_img("test_face1.jpg"))?;
    let faces = model.predict(&img)?;
    assert!(!faces.is_empty());
    assert!(faces[0].keypoints.len() > 0);
    Ok(())
}

#[test]
fn test_lpr_det_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let model = LprDetectionModel::new(
        &test_data("test_models/onnx/yolov5plate.onnx"),
        &opt,
    )?;
    let img = Image::read(&test_img("test_lpr_pipeline2.jpg"))?;
    let plates = model.predict(&img)?;
    let _ = plates;
    Ok(())
}

#[test]
fn test_lpr_rec_submodel() -> Result<()> {
    let opt = cpu_opt()?;
    let model = LprRecognizerModel::new(
        &test_data("test_models/onnx/plate_recognition_color.onnx"),
        &opt,
    )?;
    let img = Image::read(&test_img("test_lpr_recognizer.jpg"))?;
    let plates = model.predict(&img)?;
    assert!(!plates.is_empty());
    Ok(())
}

// ═══ 参数自省冒烟（直接调 ffi，不构造模型） ═══

#[test]
fn test_param_introspection_ffi() -> Result<()> {
    use modeldeploy::ffi::{self, MDModelKind, MDStatus};

    let mut names: *const libc::c_char = std::ptr::null();
    let status = unsafe { ffi::md_model_param_names(MDModelKind::DETECTION, &mut names) };
    assert_eq!(status, MDStatus::OK, "md_model_param_names failed");
    assert!(!names.is_null());
    let s = unsafe { std::ffi::CStr::from_ptr(names) }.to_string_lossy().to_string();
    assert!(s.contains("conf_threshold"), "det names should contain conf_threshold: {s}");
    assert!(s.contains("nms_threshold"), "det names should contain nms_threshold: {s}");

    let cname = std::ffi::CString::new("conf_threshold")?;
    let mut t: libc::c_char = 0;
    let status = unsafe { ffi::md_model_param_type(MDModelKind::DETECTION, cname.as_ptr(), &mut t) };
    assert_eq!(status, MDStatus::OK, "md_model_param_type failed");
    assert_eq!(t as u8 as char, 'D', "conf_threshold should be type 'D'");

    let mut sem_names: *const libc::c_char = std::ptr::null();
    let status = unsafe { ffi::md_model_param_names(MDModelKind::SEM_SEG, &mut sem_names) };
    assert_eq!(status, MDStatus::OK, "md_model_param_names (sem_seg) failed");
    let s = unsafe { std::ffi::CStr::from_ptr(sem_names) }.to_string_lossy().to_string();
    assert!(s.is_empty(), "sem_seg should have no params, got: {s}");
    Ok(())
}

// ═══ ASR / TTS（需 build_audio SDK，默认跳过） ═══

#[test]
#[ignore = "requires build_audio SDK with audio module"]
fn test_asr() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/sense_voice");
    let path = format!("{}/model.int8.onnx|{}/tokens.txt", dir, dir);
    let model = SenseVoice::new(&path, &opt)?;
    let wav = format!("{}/test_wavs/zh.wav", dir);
    let text = model.predict_wav(&wav)?;
    assert!(!text.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires build_audio SDK with audio module"]
fn test_tts() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/kokoro_v1_1");
    let path = format!("{}/model.onnx|{}/tokens.txt|{}/lexicon-gb-en.txt|{}/lexicon-zh.txt|{}/voices.bin|{}/dict|{}",
        dir, dir, dir, dir, dir, dir, dir);
    let model = Kokoro::new(&path, &opt)?;
    assert!(model.is_ready());
    Ok(())
}

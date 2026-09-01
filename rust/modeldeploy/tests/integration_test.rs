use anyhow::Result;
use modeldeploy::{
    BarcodeDetector, Classification, DbDetectorModel, DrawOptions, FaceLandmark,
    FaceRecognizerPipelineModel, FastSam, FormulaRecognizer, HandKeypoint, Image, InsightFaceAnalysis,
    InsightFaceDetModel, Kokoro, LprDetectionModel, LprPipeline, LprRecognizerModel, PaddleOCR,
    PedestrianAttribute, ReID, RecognizerModel, RuntimeOption, Scrfd, SeetaFaceAge,
    SeetaFaceGender, SeetaFaceID, SenseVoice, SpeakerGallery, SpeakerVerify, Tracker, TrackerKind,
    UltralyticsDepth, UltralyticsDet, UltralyticsObb, UltralyticsPose, UltralyticsSeg,
    UltralyticsSem, VehicleKeypoint,
};

fn test_data(rel: &str) -> String {
    format!("{}/../../test_data/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn test_img(rel: &str) -> String {
    format!("{}/../../test_data/test_images/{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn cpu_opt() -> Result<RuntimeOption> {
    let mut opt = RuntimeOption::new()?;
    opt.use_ort()
        .set_device(modeldeploy::ffi::MDDevice::CPU, 0)?
        .set_cpu_threads(4)?;
    Ok(opt)
}

// ═══ Image ═══

#[test]
fn test_image_from_device_nv12_reports_nv12_planes() -> Result<()> {
    use modeldeploy::ffi::MDDevice;
    use modeldeploy::ImageFormat;

    let width = 8;
    let height = 8;
    let mut y = vec![0u8; (width * height) as usize];
    let mut uv = vec![0u8; (width * height / 2) as usize];
    y[0] = 16;
    uv[0] = 128;

    let img = unsafe {
        Image::from_device_nv12(
            y.as_ptr(),
            uv.as_ptr(),
            width,
            height,
            width,
            width,
            MDDevice::CPU,
        )?
    };

    assert_eq!(img.format(), ImageFormat::NV12);
    assert_eq!(img.plane_count(), 2);
    assert_eq!(img.device(), MDDevice::CPU);

    let p0 = img.plane(0)?;
    assert!(!p0.data.is_null());
    let p1 = img.plane(1)?;
    assert!(!p1.data.is_null());

    assert!(img.plane(2).is_err());
    Ok(())
}

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

#[test]
fn image_to_native_bytes_len() {
    let img = Image::from_bgr24(&vec![0u8; 4 * 3 * 3], 4, 3).unwrap();
    let nb = img.to_native_bytes().unwrap();
    assert_eq!(nb.len(), 4 * 3 * 3);
}

#[test]
fn image_plane_bytes_nv12() {
    let y = vec![1u8; 4 * 2];
    let uv = vec![2u8; 4 * 1];
    let img = Image::from_nv12(&y, &uv, 4, 2, 0, 0).unwrap();
    assert_eq!(img.plane_bytes(0).unwrap().len(), 4 * 2);
    assert_eq!(img.plane_bytes(1).unwrap().len(), 4 * 1);
    assert!(img.plane_bytes(2).is_err());
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

#[test]
fn test_hand() -> Result<()> {
    let path = test_data("test_models/onnx/hand_pose.onnx");
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: hand_pose.onnx not present in test_data");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = HandKeypoint::new(&path, &opt)?;
    assert!(model.is_ready());
    let img = Image::read(&test_img("bus.jpg"))?;
    let hands = model.predict(&img)?;
    assert!(!hands.is_empty(), "should detect a hand");
    assert!(!hands[0].keypoints.is_empty());
    Ok(())
}

#[test]
fn test_vehicle_keypoint() -> Result<()> {
    let path = test_data("test_models/onnx/vehicle_keypoint.onnx");
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: vehicle_keypoint.onnx not present in test_data");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = VehicleKeypoint::new(&path, &opt)?;
    assert!(model.is_ready());
    let img = Image::read(&test_img("bus.jpg"))?;
    let r = model.predict(&img)?;
    assert!(!r.is_empty());
    assert!(r[0].keypoints.len() > 0);
    Ok(())
}

#[test]
fn test_face_landmark() -> Result<()> {
    let path = test_data("test_models/onnx/2d106det.onnx");
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: 2d106det.onnx not present in test_data");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = FaceLandmark::new(&path, &opt)?;
    assert!(model.is_ready());
    let img = Image::read(&test_img("face.jpg"))?;
    let r = model.predict(&img)?;
    assert!(!r.is_empty());
    assert_eq!(r[0].keypoints.len(), 106);
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
    let model = Scrfd::new(&test_data("test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx"), &opt)?;
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
    let model = SeetaFaceID::new(&test_data("test_models/onnx/seetaface/face_recognizer.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_id.jpg"))?;
    let rec = model.predict(&img)?;
    assert!(rec[0].embedding.len() > 0);
    Ok(())
}

// ═══ ReID（行人重识别，需 osnet_x1_0.onnx 权重；缺失则跳过） ═══

#[test]
fn test_reid() -> Result<()> {
    let model_path = test_data("test_models/onnx/osnet_x1_0.onnx");
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("OSNet model not found; skipping reid test.");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = ReID::new(&model_path, &opt)?;
    let bgr = vec![128u8; 256 * 128 * 3];
    let img = Image::from_bgr24(&bgr, 256, 128)?;
    let rec = model.predict(&img)?;
    assert_eq!(rec[0].embedding.len(), 512);
    Ok(())
}

// ═══ SpeakerVerify（声纹，需 ecapa 权重；缺失则跳过） ═══

#[test]
fn test_speaker_verify() -> Result<()> {
    let model_path = test_data("test_models/onnx/speaker/ecapa_tdnn.onnx");
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ECAPA model not found; skipping speaker verify test.");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = SpeakerVerify::new(&model_path, &opt)?;
    assert!(model.is_ready());
    let samples = vec![0.0f32; 16000];
    let emb = model.predict(&samples)?;
    assert!(!emb.is_empty(), "should produce a non-empty speaker embedding");
    Ok(())
}

// ═══ FormulaRecognizer（文档公式识别，需权重；缺失则跳过） ═══

#[test]
fn test_formula_recognizer() -> Result<()> {
    let model_path = test_data("test_models/onnx/formula_recognizer/formula_recognizer.onnx");
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("FormulaRecognizer model not found; skipping formula recognizer test.");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let model = FormulaRecognizer::new(&model_path, &opt)?;
    assert!(model.is_ready());
    let img = Image::read(&test_img("test_formula.png"))?;
    let latex = model.predict(&img)?;
    assert!(!latex.is_empty(), "should return a non-empty LaTeX string");
    Ok(())
}

// ═══ SpeakerGallery（纯内存，无权重依赖，恒定通过） ═══
#[test]
fn test_speaker_gallery() -> Result<()> {
    let mut g = SpeakerGallery::new();
    // 相互正交的向量，便于判定匹配顺序。
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let c = vec![0.0f32, 0.0, 1.0];
    g.enroll("alice", &a);
    g.enroll("bob", &b);
    g.enroll("carol", &c);
    assert_eq!(g.size(), 3);

    // 对齐 alice 的（近）同一向量 → 应把 alice 排第一，且按余弦降序。
    let matches = g.r#match(&[1.0f32, 0.05, 0.02], 3);
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].0, "alice");
    assert!(matches[0].1 >= matches[1].1);
    assert!(matches[1].1 >= matches[2].1);

    // k 截断。
    let top1 = g.r#match(&[1.0f32, 0.05, 0.02], 1);
    assert_eq!(top1.len(), 1);
    assert_eq!(top1[0].0, "alice");

    // 归一化后自身与自身余弦 = 1。
    assert!((g.r#match(&[2.0f32, 0.0, 0.0], 1)[0].1 - 1.0).abs() < 1e-5);

    // remove / clear。
    assert_eq!(g.remove("bob"), vec![true]);
    assert_eq!(g.size(), 2);
    assert_eq!(g.remove("bob"), vec![false]);
    g.clear();
    assert_eq!(g.size(), 0);
    Ok(())
}

// ═══ FaceAge / Gender ═══

#[test]
fn test_face_age_gender() -> Result<()> {
    let opt = cpu_opt()?;
    let age_model = SeetaFaceAge::new(&test_data("test_models/onnx/seetaface/age_predictor.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_id1.jpg"))?;
    let age: i32 = age_model.predict(&img)?;
    assert!(age >= 0);

    let gender_model = SeetaFaceGender::new(&test_data("test_models/onnx/seetaface/gender_predictor.onnx"), &opt)?;
    let img2 = Image::read(&test_img("test_face_gender.jpg"))?;
    let gender: i32 = gender_model.predict(&img2)?;
    assert!(gender >= 0);
    Ok(())
}

// ═══ 标量类型断言：SeetaFaceAge/Gender predict 返回 i32（编译期验证，不加载模型） ═══

#[test]
fn face_age_gender_returns_i32() -> Result<()> {
    let opt = cpu_opt()?;
    let age_model = SeetaFaceAge::new(&test_data("test_models/onnx/seetaface/age_predictor.onnx"), &opt)?;
    let img = Image::read(&test_img("test_face_id1.jpg"))?;

    let mut _a: i32 = 0;
    _a = age_model.predict(&img)?;
    let _: Result<i32, modeldeploy::MdError> = age_model.predict(&img);

    let gender_model = SeetaFaceGender::new(&test_data("test_models/onnx/seetaface/gender_predictor.onnx"), &opt)?;
    let img2 = Image::read(&test_img("test_face_gender.jpg"))?;
    let mut _g: i32 = 0;
    _g = gender_model.predict(&img2)?;
    let _: Result<i32, modeldeploy::MdError> = gender_model.predict(&img2);
    Ok(())
}

// ═══ 批量推理 predict_batch ═══

#[test]
fn test_detection_predict_batch() -> Result<()> {
    let opt = cpu_opt()?;
    let model = UltralyticsDet::new(&test_data("test_models/onnx/yolo11n/yolo11n.onnx"), &opt)?;
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let imgs = [&img, &img];
    let dets = model.predict_batch(&imgs)?;
    // 2D：按图返回，两图各一组
    assert_eq!(dets.len(), 2);
    for group in &dets {
        assert!(!group.is_empty());
        assert!(group[0].score > 0.0 && group[0].score <= 1.0);
        assert!(group[0].rect.width > 0.0);
    }
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
    // OCR 外部控制：max_side_len / cls_batch / rec_batch / rec_image_shape
    model.set_max_side_len(960)?;
    model.set_cls_batch_size(2)?;
    model.set_rec_batch_size(-1)?;
    model.set_rec_image_shape(3, 48, 320)?;
    assert!(model.set_cls_batch_size(0).is_err());
    assert!(model.set_rec_batch_size(-2).is_err());
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
    // cls batch size：-1 自动、>0 固定；0 / <-1 非法
    model.set_cls_batch_size(-1)?;
    model.set_cls_batch_size(2)?;
    assert!(model.set_cls_batch_size(0).is_err());
    assert!(model.set_cls_batch_size(-2).is_err());
    model.set_cls_batch_size(1)?;
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
    let dir = test_data("test_models/onnx/seetaface");
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
    // 类型化 setter：阈值 + 关键点 + 输入尺寸
    model.set_conf_threshold(0.35)?;
    model.set_nms_threshold(0.5)?;
    model.set_landmarks_per_card(4.0)?;
    model.set_input_size(640, 640)?;
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
fn test_asr_structured() -> Result<()> {
    let opt = cpu_opt()?;
    let dir = test_data("test_models/onnx/sense_voice");
    let path = format!("{}/model.int8.onnx|{}/tokens.txt", dir, dir);
    let model = SenseVoice::new(&path, &opt)?;
    let wav = format!("{}/test_wavs/zh.wav", dir);
    let r = model.predict_wav_structured(&wav)?;
    assert!(!r.text.is_empty());
    assert_eq!(r.language, "zh");
    assert_eq!(r.event, "Speech");
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

// ═══ 多目标跟踪器（纯 CPU 无模型依赖） ═══

#[test]
fn test_tracker_byte_track_stable_id() -> Result<()> {
    use modeldeploy::Rect;

    let mut tracker = Tracker::new(TrackerKind::ByteTrack)?;
    assert_eq!(tracker.kind(), TrackerKind::ByteTrack);

    let boxes = vec![Rect { x: 100.0, y: 100.0, width: 50.0, height: 100.0 }];
    let scores = vec![0.9f32];
    let label_ids = vec![0i32];

    // 帧 1：新出现目标
    let frame1 = tracker.update(&boxes, &scores, &label_ids)?;
    assert!(!frame1.is_empty(), "frame1 应产生跟踪目标");
    let id1 = frame1[0].track_id;

    // 帧 2：目标轻微移动，ID 应保持稳定
    let boxes2 = vec![Rect { x: 112.0, y: 104.0, width: 50.0, height: 100.0 }];
    let frame2 = tracker.update(&boxes2, &scores, &label_ids)?;
    assert!(!frame2.is_empty(), "frame2 应产生跟踪目标");
    let id2 = frame2[0].track_id;

    assert_eq!(id1, id2, "跟踪 ID 跨帧应稳定");
    Ok(())
}

// ═══ 条码 / 二维码解码（纯 CPU 无模型依赖） ═══

#[test]
fn test_barcode_qr_decode() -> Result<()> {
    let det = BarcodeDetector::new()?;
    let img = Image::read(&test_data("qr_sample.png"))?;
    let res = det.detect(&img)?;
    assert!(!res.is_empty(), "QR 样本应至少解码出一个码");
    assert!(res[0].is_qr, "应为二维码");
    assert_eq!(res[0].text, "https://example.com/MD");
    Ok(())
}

#[test]
fn test_tracker_set_param_and_reset() -> Result<()> {
    use modeldeploy::Rect;

    let mut tracker = Tracker::new(TrackerKind::ByteTrack)?;
    // 命名参数设置
    tracker.set_param("track_thresh", 0.5)?;
    tracker.set_param("max_age", 30.0)?;
    // 未知名参数应报错
    assert!(tracker.set_param("no_such_param", 1.0).is_err());

    let boxes = vec![Rect { x: 10.0, y: 10.0, width: 40.0, height: 80.0 }];
    let scores = vec![0.9f32];
    let label_ids = vec![0i32];
    let f1 = tracker.update(&boxes, &scores, &label_ids)?;
    assert!(!f1.is_empty());

    // reset 后仍能继续跟踪
    tracker.reset()?;
    let f2 = tracker.update(&boxes, &scores, &label_ids)?;
    assert!(!f2.is_empty());
    Ok(())
}

// F1 回归：update() 遵循 查询(capacity,非变异)→分配→单次提交 契约，每逻辑帧只推进一次。
// 若仍用 update 自身做两阶段探测（双重推进），一个 Lost 目标的 max_age 会按每帧 +2 增长、
// 有效减半并被过早移除；本测试会让 A 在 max_age 内于"另一目标持续出现"的环境下重新出现，
// 断言其 ID 保持不变 —— 修复前此处失败（A 已丢、得到新 id）。
#[test]
fn test_tracker_no_double_advance_keeps_lost_id() -> Result<()> {
    use modeldeploy::Rect;

    let mut tracker = Tracker::new(TrackerKind::ByteTrack)?;
    tracker.set_param("max_age", 30.0)?;

    // 帧 1、2：目标 A
    let a = Rect { x: 100.0, y: 100.0, width: 40.0, height: 40.0 };
    let a2 = Rect { x: 102.0, y: 102.0, width: 40.0, height: 40.0 };
    let s = vec![0.95f32];
    let l = vec![0i32];
    let frame1 = tracker.update(&vec![a.clone()], &s, &l)?;
    assert!(!frame1.is_empty(), "frame1 应产生跟踪目标");
    let a_id = frame1[0].track_id;
    let _ = tracker.update(&vec![a2], &s, &l)?;

    // 帧 3..25：A 消失，目标 B 持续出现于远处（每帧有输出）
    let b = Rect { x: 400.0, y: 400.0, width: 40.0, height: 40.0 };
    let sb = vec![0.95f32];
    let lb = vec![1i32];
    for _ in 0..23 {
        let r = tracker.update(&vec![b.clone()], &sb, &lb)?;
        assert!(!r.is_empty(), "B 每帧都应被跟踪");
    }

    // 帧 26：A 重新出现（连同 B）—— 修复前 A 已被移除（新 id），修复后仍为原 id
    let s2 = vec![0.95f32, 0.95f32];
    let l2 = vec![0i32, 1i32];
    let r = tracker.update(&vec![a, b], &s2, &l2)?;
    let track_a = r.iter().find(|t| t.rect.x == 100.0 && t.rect.y == 100.0);
    assert!(track_a.is_some(), "A 重新出现时应被跟踪");
    assert_eq!(
        track_a.unwrap().track_id,
        a_id,
        "A 的 ID 应保持不变（无双重推进）"
    );
    Ok(())
}

// ═══ CV 解决方案（无权重） ═══

#[test]
fn test_cv_solution() -> Result<()> {
    use modeldeploy::{vision_iou, ObjectCounter};
    let counter = ObjectCounter::new()?;
    counter.set_line((5.0, 0.0), (5.0, 10.0))?;
    // 框中心 x=1（线左）
    counter.update(&[0.0, 4.0, 2.0, 2.0], &[0], &[1])?;
    let (i0, _o0) = counter.hline()?;
    assert_eq!(i0, 0);
    // 框中心 x=9（线右）
    counter.update(&[8.0, 4.0, 2.0, 2.0], &[0], &[1])?;
    let (i1, _o1) = counter.hline()?;
    assert_eq!(i1, 1);
    let iou = vision_iou(0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0)?;
    assert!((iou - 1.0).abs() < 1e-5);
    Ok(())
}

// ═══ 区域解决方案（无权重） ═══

#[test]
fn region_solutions_count() -> Result<()> {
    use modeldeploy::{QueueManager, RegionCounter, TrackZone};
    let poly = [0f32, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0];
    let boxes = [2f32, 2.0, 2.0, 2.0, 50.0, 50.0, 2.0, 2.0];
    let ids = [1i32, 2];
    let labs = [0i32, 0];

    let rc = RegionCounter::new()?;
    rc.add_region("A", &poly)?;
    rc.update(&boxes, &ids, &labs)?;
    assert_eq!(rc.count("A"), 1);

    let q = QueueManager::new()?;
    q.set_region(&poly)?;
    q.update(&boxes, &ids, &labs)?;
    assert_eq!(q.count(), 1);

    let tz = TrackZone::new()?;
    tz.set_region(&poly)?;
    tz.update(&boxes, &ids, &labs)?;
    assert_eq!(tz.count(), 1);
    Ok(())
}

// ═══ 音频解决方案（无权重） ═══

#[test]
fn test_audio_speaker_search() -> Result<()> {
    use modeldeploy::{resample, SpeakerSearch};
    let search = SpeakerSearch::new()?;
    search.enroll("alice", &[1.0, 0.0, 0.0])?;
    let best = search.match_top(&[0.99, 0.1, 0.0])?;
    assert_eq!(best, "alice");
    let out = resample(&vec![0.0f32; 800], 8000, 16000)?;
    assert!(out.len() == 1600);
    Ok(())
}

// ═══ NLP 工具（无权重） ═══

#[test]
fn test_nlp_tools() -> Result<()> {
    use modeldeploy::{nlp_split_sentences, nlp_stats};
    let sents = nlp_split_sentences("hello world")?;
    assert_eq!(sents, vec!["hello world"]);
    let (chars, words, _sents) = nlp_stats("a b c")?;
    assert_eq!(words, 3);
    assert_eq!(chars, 5);
    Ok(())
}

// ═══ FastSam（交互式分割提示） ═══

#[test]
fn fastsam_prompt() -> Result<()> {
    let path = test_data("test_models/onnx/FastSAM-s.onnx");
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: FastSAM-s.onnx not present in test_data");
        return Ok(());
    }
    let opt = cpu_opt()?;
    let m = FastSam::new(&path, &opt)?;
    let img = Image::read(&test_img("test_detection0.jpg"))?;
    let all = m.predict(&img)?;
    if all.is_empty() {
        return Ok(());
    }
    let r = &all[0].rect;
    let b = [r.x, r.y, r.width, r.height];
    let sel = m.predict_with_prompts(&img, &b, &[], &[])?;
    assert!(!sel.is_empty() && sel.len() <= all.len());
    Ok(())
}

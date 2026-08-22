use crate::error::{check_status, MdError};
use crate::ffi;
use crate::image::{cstr_to_string, Image};
use crate::runtime::RuntimeOption;
use crate::types::*;
use std::ffi::CString;
use std::ptr;

/// 统一模型载体（capi 单一分发内部实现；各具体模型包装它）
pub struct Model {
    pub(crate) handle: ffi::MDModelHandle,
    pub(crate) kind: ModelKind,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// 创建模型。多子模型路径用 '|' 分隔（OCR/LPR pipeline/insightface/ASR/TTS）。
    pub fn new(kind: ModelKind, model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        let cpath = CString::new(model_path)
            .map_err(|_| MdError::InvalidArgument("model_path".into()))?;
        let mut handle = ptr::null_mut();
        let status = unsafe {
            ffi::md_model_create(&mut handle, kind.to_ffi(), cpath.as_ptr(), option.handle)
        };
        check_status(status)?;
        if handle.is_null() {
            return Err(MdError::ModelInit(model_path.into()));
        }
        Ok(Self { handle, kind })
    }

    /// 深拷贝（复用已加载的 backend session）
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, MdError> {
        let mut out = ptr::null_mut();
        check_status(unsafe { ffi::md_model_clone(self.handle, &mut out) })?;
        Ok(Self {
            handle: out,
            kind: self.kind,
        })
    }

    pub fn is_ready(&self) -> bool {
        unsafe { ffi::md_model_ready(self.handle) == ffi::MDStatus::OK }
    }

    /// 设置输入尺寸（pipeline 模型设置检测子模型尺寸）
    pub fn set_input_size(&self, w: i32, h: i32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_model_set_input_size(self.handle, w, h) })
    }

    /// 设置 pipeline 分类子模型输入尺寸（PedestrianAttribute）
    pub fn set_cls_input_size(&self, w: i32, h: i32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_model_set_cls_input_size(self.handle, w, h) })
    }

    /// set pipeline cls-submodel batch size (>0 fixed, -1 auto; PedestrianAttribute).
    /// For Sophgo int8 bmodel (batch=1 static shape) set to 1.
    pub fn set_cls_batch_size(&self, batch: i32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_model_set_cls_batch_size(self.handle, batch) })
    }

    /// set OCR rec-submodel batch size (>0 fixed, -1 auto).
    pub fn set_rec_batch_size(&self, batch: i32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_model_set_rec_batch_size(self.handle, batch) })
    }

    /// set OCR rec-submodel input shape (c, h, w).
    pub fn set_rec_image_shape(&self, c: i32, h: i32, w: i32) -> Result<(), MdError> {
        check_status(unsafe {
            ffi::md_model_set_rec_image_shape(self.handle, c, h, w)
        })
    }

    /// 按扁平参数名设置模型前/后处理参数（整型）。
    pub fn set_param_int(&self, name: &str, value: i64) -> Result<(), MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        check_status(unsafe { ffi::md_model_set_param_i(self.handle, cn.as_ptr(), value) })
    }

    /// 按扁平参数名设置模型前/后处理参数（浮点）。
    pub fn set_param_double(&self, name: &str, value: f64) -> Result<(), MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        check_status(unsafe { ffi::md_model_set_param_d(self.handle, cn.as_ptr(), value) })
    }

    /// 按扁平参数名设置模型前/后处理参数（布尔）。
    pub fn set_param_bool(&self, name: &str, value: bool) -> Result<(), MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        check_status(unsafe {
            ffi::md_model_set_param_b(self.handle, cn.as_ptr(), value as libc::c_int)
        })
    }

    /// 按扁平参数名设置模型前/后处理参数（字符串/枚举）。
    pub fn set_param_str(&self, name: &str, value: &str) -> Result<(), MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        let cv = CString::new(value).map_err(|_| MdError::InvalidArgument("value".into()))?;
        check_status(unsafe { ffi::md_model_set_param_s(self.handle, cn.as_ptr(), cv.as_ptr()) })
    }

    /// 该模型 kind 支持的参数名（kind 级自省；无需模型就绪）。
    pub fn param_names(&self) -> Result<Vec<String>, MdError> {
        let mut p: *const libc::c_char = ptr::null();
        check_status(unsafe { ffi::md_model_param_names(self.kind.to_ffi(), &mut p) })?;
        if p.is_null() {
            return Ok(Vec::new());
        }
        let s = unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned();
        Ok(s.split('|')
            .filter(|x| !x.is_empty())
            .map(|x| x.to_string())
            .collect())
    }

    /// 该模型 kind 下某参数的类型字符（'I'/'D'/'B'/'S'）。
    pub fn param_type(&self, name: &str) -> Result<char, MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        let mut t: libc::c_char = 0;
        check_status(unsafe { ffi::md_model_param_type(self.kind.to_ffi(), cn.as_ptr(), &mut t) })?;
        Ok(t as u8 as char)
    }

    /// 检测/关键点/实例分割通用：置信度阈值（detection/obb/pose/iseg/facedet 等）。
    pub fn set_conf_threshold(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("conf_threshold", v)
    }

    /// 检测/关键点/实例分割通用：NMS 阈值（detection/obb/pose/iseg/facedet 等）。
    pub fn set_nms_threshold(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("nms_threshold", v)
    }

    /// 姿态：关键点数量。
    pub fn set_keypoints_num(&self, v: i64) -> Result<(), MdError> {
        self.set_param_int("keypoints_num", v)
    }

    /// 实例分割：掩码二值化阈值。
    pub fn set_mask_threshold(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("mask_threshold", v)
    }

    /// 分类：Top-K 输出个数。
    pub fn set_top_k(&self, v: i64) -> Result<(), MdError> {
        self.set_param_int("top_k", v)
    }

    /// 分类：是否多标签。
    pub fn set_multi_label(&self, v: bool) -> Result<(), MdError> {
        self.set_param_bool("multi_label", v)
    }

    /// 人脸检测：每人脸关键点数量。
    pub fn set_landmarks_per_face(&self, v: i64) -> Result<(), MdError> {
        self.set_param_int("landmarks_per_face", v)
    }

    /// OCR：DB 检测二值化阈值。
    pub fn set_det_db_thresh(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("det_db_thresh", v)
    }

    /// OCR：DB 检测框阈值。
    pub fn set_det_db_box_thresh(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("det_db_box_thresh", v)
    }

    /// OCR：DB 检测 unclip 比率。
    pub fn set_det_db_unclip_ratio(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("det_db_unclip_ratio", v)
    }

    /// OCR：DB 检测得分模式（枚举字符串）。
    pub fn set_det_db_score_mode(&self, v: &str) -> Result<(), MdError> {
        self.set_param_str("det_db_score_mode", v)
    }

    /// OCR：是否启用膨胀。
    pub fn set_use_dilation(&self, v: bool) -> Result<(), MdError> {
        self.set_param_bool("use_dilation", v)
    }

    /// OCR：方向分类阈值。
    pub fn set_cls_thresh(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("cls_thresh", v)
    }

    /// 行人属性：行人检测阈值。
    pub fn set_det_threshold(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("det_threshold", v)
    }

    /// insightface：人脸检测阈值。
    pub fn set_det_thresh(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("det_thresh", v)
    }

    /// OCR：DB 检测最长边（缩放主控）。
    pub fn set_max_side_len(&self, v: i64) -> Result<(), MdError> {
        self.set_param_int("max_side_len", v)
    }

    /// 车牌检测：每车牌关键点数量。
    pub fn set_landmarks_per_card(&self, v: f64) -> Result<(), MdError> {
        self.set_param_double("landmarks_per_card", v)
    }

    /// 推理：返回持句柄的结果包装
    pub fn predict(&self, image: &Image) -> Result<RawResult, MdError> {
        let mut result = ptr::null_mut();
        check_status(unsafe { ffi::md_model_predict(self.handle, image.handle, &mut result) })?;
        if result.is_null() {
            return Err(MdError::ModelPredict("null result".into()));
        }
        Ok(RawResult {
            handle: result,
            _kind: self.kind,
        })
    }

    /// 批量推理（多图一次提交，结果平铺），返回 RawResult（调用方用对应 reader 读取）
    pub fn predict_batch(&self, images: &[&Image]) -> Result<RawResult, MdError> {
        let mut handles: Vec<ffi::MDImageHandle> = images.iter().map(|im| im.handle).collect();
        let mut result = ptr::null_mut();
        check_status(unsafe {
            ffi::md_model_predict_batch(self.handle, handles.as_mut_ptr(), handles.len(), &mut result)
        })?;
        if result.is_null() {
            return Err(MdError::ModelPredict("null batch result".into()));
        }
        Ok(RawResult {
            handle: result,
            _kind: self.kind,
        })
    }

    /// ASR：从 wav 文件识别
    pub fn asr_wav(&self, wav_path: &str) -> Result<String, MdError> {
        let cpath = CString::new(wav_path).map_err(|_| MdError::InvalidArgument("wav".into()))?;
        let mut text: *const libc::c_char = ptr::null();
        check_status(unsafe { ffi::md_audio_asr_wav(self.handle, cpath.as_ptr(), &mut text) })?;
        Ok(unsafe { cstr_to_string(text) })
    }

    /// ASR：从 PCM float 采样识别
    pub fn asr(&self, samples: &[f32], sample_rate: i32) -> Result<String, MdError> {
        let mut text: *const libc::c_char = ptr::null();
        check_status(unsafe {
            ffi::md_audio_asr(self.handle, samples.as_ptr(), samples.len(), sample_rate, &mut text)
        })?;
        Ok(unsafe { cstr_to_string(text) })
    }

    /// TTS：文本合成音频
    pub fn tts(&self, text: &str, voice: &str, speed: f32) -> Result<TtsAudio, MdError> {
        let ctext = CString::new(text).map_err(|_| MdError::InvalidArgument("text".into()))?;
        let cvoice = CString::new(voice).map_err(|_| MdError::InvalidArgument("voice".into()))?;
        let mut sample_rate = 0;
        let mut audio: *const f32 = ptr::null();
        let mut n = 0usize;
        check_status(unsafe {
            ffi::md_audio_tts(
                self.handle,
                ctext.as_ptr(),
                cvoice.as_ptr(),
                speed,
                &mut sample_rate,
                &mut audio,
                &mut n,
            )
        })?;
        let samples = if n > 0 && !audio.is_null() {
            unsafe { std::slice::from_raw_parts(audio, n) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(TtsAudio {
            samples,
            sample_rate,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_model_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 原始结果句柄包装：持句柄，可读取各类型结果 / 绘制，Drop 释放
pub struct RawResult {
    pub(crate) handle: ffi::MDResultHandle,
    pub(crate) _kind: ModelKind,
}

unsafe impl Send for RawResult {}
unsafe impl Sync for RawResult {}

impl RawResult {
    /// 结果实例数
    pub fn count(&self) -> Result<usize, MdError> {
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_count(self.handle, &mut n) })?;
        Ok(n)
    }

    pub fn detection(&self) -> Result<Vec<Detection>, MdError> {
        let mut items: *const ffi::MDDetectionItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_detection(self.handle, &mut items, &mut n) })?;
        Ok(slice_items(items, n)
            .iter()
            .map(|it| Detection {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                label_id: it.label_id,
                score: it.score,
            })
            .collect())
    }

    pub fn classification(&self) -> Result<Vec<ClassificationResult>, MdError> {
        let mut items: *const ffi::MDClassifyItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_classification(self.handle, &mut items, &mut n) })?;
        Ok(slice_items(items, n)
            .iter()
            .map(|it| ClassificationResult {
                label_id: it.label_id,
                score: it.score,
            })
            .collect())
    }

    pub fn pose(&self) -> Result<Vec<Pose>, MdError> {
        let mut items: *const ffi::MDPoseItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_pose(self.handle, &mut items, &mut n) })?;
        let items = slice_items(items, n);
        let mut out = Vec::with_capacity(n);
        for (i, it) in items.iter().enumerate() {
            let mut kps: *const ffi::MDPoint3 = ptr::null();
            let mut kn = 0usize;
            check_status(unsafe { ffi::md_result_keypoints(self.handle, i, &mut kps, &mut kn) })?;
            out.push(Pose {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                score: it.score,
                keypoints: slice_items(kps, kn)
                    .iter()
                    .map(|p| Point3 {
                        x: p.x,
                        y: p.y,
                        z: p.z,
                    })
                    .collect(),
            });
        }
        Ok(out)
    }

    pub fn obb(&self) -> Result<Vec<Obb>, MdError> {
        let mut items: *const ffi::MDObbItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_obb(self.handle, &mut items, &mut n) })?;
        Ok(slice_items(items, n)
            .iter()
            .map(|it| Obb {
                rotated_box: RotatedBox {
                    cx: it.cx,
                    cy: it.cy,
                    width: it.w,
                    height: it.h,
                    angle: it.angle,
                },
                label_id: it.label_id,
                score: it.score,
            })
            .collect())
    }

    pub fn instance_seg(&self) -> Result<Vec<InstanceSeg>, MdError> {
        let mut items: *const ffi::MDIsegItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_instance_seg(self.handle, &mut items, &mut n) })?;
        Ok(slice_items(items, n)
            .iter()
            .map(|it| InstanceSeg {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                label_id: it.label_id,
                score: it.score,
            })
            .collect())
    }

    pub fn sem_seg(&self) -> Result<SemSeg, MdError> {
        let mut labels: *const u8 = ptr::null();
        let mut h = 0usize;
        let mut w = 0usize;
        let mut nc = 0;
        check_status(unsafe {
            ffi::md_result_sem_seg(self.handle, &mut labels, &mut h, &mut w, &mut nc)
        })?;
        Ok(SemSeg {
            labels: read_bytes(labels, h * w),
            height: h,
            width: w,
            num_classes: nc,
        })
    }

    pub fn depth(&self) -> Result<Depth, MdError> {
        let mut depth: *const f32 = ptr::null();
        let mut h = 0usize;
        let mut w = 0usize;
        check_status(unsafe { ffi::md_result_depth(self.handle, &mut depth, &mut h, &mut w) })?;
        Ok(Depth {
            depth: read_f32(depth, h * w),
            height: h,
            width: w,
        })
    }

    pub fn face_det(&self) -> Result<Vec<FaceDetection>, MdError> {
        let mut items: *const ffi::MDFaceItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_face(self.handle, &mut items, &mut n) })?;
        let items = slice_items(items, n);
        let mut out = Vec::with_capacity(n);
        for (i, it) in items.iter().enumerate() {
            let mut kps: *const ffi::MDPoint = ptr::null();
            let mut kn = 0usize;
            check_status(unsafe { ffi::md_result_face_kps(self.handle, i, &mut kps, &mut kn) })?;
            out.push(FaceDetection {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                score: it.score,
                keypoints: slice_items(kps, kn)
                    .iter()
                    .map(|p| Point { x: p.x, y: p.y })
                    .collect(),
            });
        }
        Ok(out)
    }

    pub fn face_recognition(&self, i: usize) -> Result<FaceRecognition, MdError> {
        let mut emb: *const f32 = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_face_embedding(self.handle, i, &mut emb, &mut n) })?;
        Ok(FaceRecognition {
            embedding: read_f32(emb, n),
        })
    }

    /// 读取所有人脸 embedding（face-rec pipeline）
    pub fn face_recognition_all(&self) -> Result<Vec<FaceRecognition>, MdError> {
        let mut out = Vec::new();
        let n = self.count()?;
        for i in 0..n {
            out.push(self.face_recognition(i)?);
        }
        Ok(out)
    }

    pub fn reid(&self, i: usize) -> Result<ReIdResult, MdError> {
        let mut emb: *const f32 = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_reid_embedding(self.handle, i, &mut emb, &mut n) })?;
        Ok(ReIdResult {
            embedding: read_f32(emb, n),
        })
    }

    /// 读取全部 ReID embedding
    pub fn reid_all(&self) -> Result<Vec<ReIdResult>, MdError> {
        let mut out = Vec::new();
        let n = self.count()?;
        for i in 0..n {
            out.push(self.reid(i)?);
        }
        Ok(out)
    }

    pub fn insightface(&self) -> Result<Vec<InsightFace>, MdError> {
        let mut items: *const ffi::MDInsightFaceItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_insightface(self.handle, &mut items, &mut n) })?;
        let items = slice_items(items, n);
        let mut out = Vec::with_capacity(n);
        for (i, it) in items.iter().enumerate() {
            let mut kps: *const ffi::MDPoint = ptr::null();
            let mut kn = 0usize;
            let mut emb: *const f32 = ptr::null();
            let mut en = 0usize;
            let mut pose: *const f32 = ptr::null();
            let mut pn = 0usize;
            check_status(unsafe { ffi::md_result_insightface_kps(self.handle, i, &mut kps, &mut kn) })?;
            check_status(unsafe { ffi::md_result_insightface_embedding(self.handle, i, &mut emb, &mut en) })?;
            check_status(unsafe { ffi::md_result_insightface_pose(self.handle, i, &mut pose, &mut pn) })?;
            out.push(InsightFace {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                score: it.score,
                keypoints: slice_items(kps, kn)
                    .iter()
                    .map(|p| Point { x: p.x, y: p.y })
                    .collect(),
                embedding: read_f32(emb, en),
                pose: read_f32(pose, pn),
                gender: it.gender,
                age: it.age,
            });
        }
        Ok(out)
    }

    pub fn ocr(&self) -> Result<Vec<OcrLine>, MdError> {
        let n = self.count()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut quad: *const i32 = ptr::null();
            let mut text: *const libc::c_char = ptr::null();
            let mut score = 0f32;
            let status = unsafe { ffi::md_result_ocr(self.handle, i, &mut quad, &mut text, &mut score) };
            if status != ffi::MDStatus::OK {
                break;
            }
            let mut q = [0i32; 8];
            if !quad.is_null() {
                let s = unsafe { std::slice::from_raw_parts(quad, 8) };
                q.copy_from_slice(s);
            }
            out.push(OcrLine {
                quad: q,
                text: unsafe { cstr_to_string(text) },
                score,
                cls_label: 0,
                cls_score: 0.0,
            });
        }
        // 补充 cls_label / cls_score（OCR 方向分类）
        for (idx, line) in out.iter_mut().enumerate() {
            let mut cls_label = 0;
            let mut cls_score = 0f32;
            let status = unsafe {
                ffi::md_result_ocr_cls(self.handle, idx, &mut cls_label, &mut cls_score)
            };
            if status == ffi::MDStatus::OK {
                line.cls_label = cls_label;
                line.cls_score = cls_score;
            }
        }
        Ok(out)
    }

    pub fn lpr(&self) -> Result<Vec<LicensePlate>, MdError> {
        let mut items: *const ffi::MDLprItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_lpr(self.handle, &mut items, &mut n) })?;
        let items = slice_items(items, n);
        let mut out = Vec::with_capacity(n);
        for (i, it) in items.iter().enumerate() {
            let mut plate: *const libc::c_char = ptr::null();
            let mut color: *const libc::c_char = ptr::null();
            let mut kps: *const ffi::MDPoint = ptr::null();
            let mut kn = 0usize;
            check_status(unsafe { ffi::md_result_plate(self.handle, i, &mut plate, &mut color) })?;
            check_status(unsafe { ffi::md_result_lpr_keypoints(self.handle, i, &mut kps, &mut kn) })?;
            out.push(LicensePlate {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                plate: unsafe { cstr_to_string(plate) },
                color: unsafe { cstr_to_string(color) },
                score: it.score,
                keypoints: slice_items(kps, kn)
                    .iter()
                    .map(|p| Point { x: p.x, y: p.y })
                    .collect(),
            });
        }
        Ok(out)
    }

    pub fn attribute(&self) -> Result<Vec<Attribute>, MdError> {
        let mut items: *const ffi::MDAttrItem = ptr::null();
        let mut n = 0usize;
        check_status(unsafe { ffi::md_result_attribute(self.handle, &mut items, &mut n) })?;
        let items = slice_items(items, n);
        let mut out = Vec::with_capacity(n);
        for (i, it) in items.iter().enumerate() {
            let mut scores: *const f32 = ptr::null();
            let mut sn = 0usize;
            check_status(unsafe { ffi::md_result_attr_scores(self.handle, i, &mut scores, &mut sn) })?;
            out.push(Attribute {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                box_label_id: it.box_label_id,
                box_score: it.box_score,
                attr_scores: read_f32(scores, sn),
            });
        }
        Ok(out)
    }

    pub fn age(&self) -> Result<i32, MdError> {
        let mut age = 0;
        check_status(unsafe { ffi::md_result_age(self.handle, &mut age) })?;
        Ok(age)
    }

    pub fn gender(&self) -> Result<i32, MdError> {
        let mut g = 0;
        check_status(unsafe { ffi::md_result_gender(self.handle, &mut g) })?;
        Ok(g)
    }

    // ────────────────── 2D 批量结果（按图索引 img，逐图取项数组） ──────────────────
    pub fn detection_batch(&self) -> Result<Vec<Vec<Detection>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDDetectionItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_detection_batch(self.handle, img, &mut items, &mut n) })?;
            out.push(slice_items(items, n)
                .iter()
                .map(|it| Detection {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    label_id: it.label_id,
                    score: it.score,
                })
                .collect());
        }
        Ok(out)
    }

    pub fn classification_batch(&self) -> Result<Vec<Vec<ClassificationResult>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDClassifyItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_classification_batch(self.handle, img, &mut items, &mut n) })?;
            out.push(slice_items(items, n)
                .iter()
                .map(|it| ClassificationResult { label_id: it.label_id, score: it.score })
                .collect());
        }
        Ok(out)
    }

    pub fn pose_batch(&self) -> Result<Vec<Vec<Pose>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDPoseItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_pose_batch(self.handle, img, &mut items, &mut n) })?;
            let items = slice_items(items, n);
            let mut per_img = Vec::with_capacity(n);
            for (i, it) in items.iter().enumerate() {
                let mut kps: *const ffi::MDPoint3 = ptr::null();
                let mut kn = 0usize;
                check_status(unsafe { ffi::md_result_keypoints_batch(self.handle, img, i, &mut kps, &mut kn) })?;
                per_img.push(Pose {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    score: it.score,
                    keypoints: slice_items(kps, kn)
                        .iter()
                        .map(|p| Point3 { x: p.x, y: p.y, z: p.z })
                        .collect(),
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    pub fn obb_batch(&self) -> Result<Vec<Vec<Obb>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDObbItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_obb_batch(self.handle, img, &mut items, &mut n) })?;
            out.push(slice_items(items, n)
                .iter()
                .map(|it| Obb {
                    rotated_box: RotatedBox { cx: it.cx, cy: it.cy, width: it.w, height: it.h, angle: it.angle },
                    label_id: it.label_id,
                    score: it.score,
                })
                .collect());
        }
        Ok(out)
    }

    pub fn instance_seg_batch(&self) -> Result<Vec<Vec<InstanceSeg>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDIsegItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_instance_seg_batch(self.handle, img, &mut items, &mut n) })?;
            out.push(slice_items(items, n)
                .iter()
                .map(|it| InstanceSeg {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    label_id: it.label_id,
                    score: it.score,
                })
                .collect());
        }
        Ok(out)
    }

    /// 批量语义分割：每图一个 SemSeg（域与单图保持一致）。
    pub fn sem_seg_batch(&self) -> Result<Vec<Vec<SemSeg>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut labels: *const u8 = ptr::null();
            let mut h = 0usize;
            let mut w = 0usize;
            let mut nc = 0;
            check_status(unsafe { ffi::md_result_sem_seg_batch(self.handle, img, &mut labels, &mut h, &mut w, &mut nc) })?;
            out.push(vec![SemSeg { labels: read_bytes(labels, h * w), height: h, width: w, num_classes: nc }]);
        }
        Ok(out)
    }

    /// 批量深度：每图一个 Depth（域与单图保持一致）。
    pub fn depth_batch(&self) -> Result<Vec<Vec<Depth>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut depth: *const f32 = ptr::null();
            let mut h = 0usize;
            let mut w = 0usize;
            check_status(unsafe { ffi::md_result_depth_batch(self.handle, img, &mut depth, &mut h, &mut w) })?;
            out.push(vec![Depth { depth: read_f32(depth, h * w), height: h, width: w }]);
        }
        Ok(out)
    }

    pub fn face_det_batch(&self) -> Result<Vec<Vec<FaceDetection>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDFaceItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_face_batch(self.handle, img, &mut items, &mut n) })?;
            let items = slice_items(items, n);
            let mut per_img = Vec::with_capacity(n);
            for (i, it) in items.iter().enumerate() {
                let mut kps: *const ffi::MDPoint = ptr::null();
                let mut kn = 0usize;
                check_status(unsafe { ffi::md_result_face_kps_batch(self.handle, img, i, &mut kps, &mut kn) })?;
                per_img.push(FaceDetection {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    score: it.score,
                    keypoints: slice_items(kps, kn)
                        .iter()
                        .map(|p| Point { x: p.x, y: p.y })
                        .collect(),
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    /// 批量人脸识别（face-rec / pipeline）：每图一个 embedding。
    pub fn face_recognition_batch(&self) -> Result<Vec<Vec<FaceRecognition>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut emb: *const f32 = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_face_embedding_batch(self.handle, img, &mut emb, &mut n) })?;
            out.push(vec![FaceRecognition { embedding: read_f32(emb, n) }]);
        }
        Ok(out)
    }

    /// 批量行人重识别：每图一个 embedding。
    pub fn reid_batch(&self) -> Result<Vec<Vec<ReIdResult>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut emb: *const f32 = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_reid_embedding_batch(self.handle, img, &mut emb, &mut n) })?;
            out.push(vec![ReIdResult { embedding: read_f32(emb, n) }]);
        }
        Ok(out)
    }

    pub fn insightface_batch(&self) -> Result<Vec<Vec<InsightFace>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDInsightFaceItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_insightface_batch(self.handle, img, &mut items, &mut n) })?;
            let items = slice_items(items, n);
            let mut per_img = Vec::with_capacity(n);
            for (i, it) in items.iter().enumerate() {
                let mut kps: *const ffi::MDPoint = ptr::null();
                let mut kn = 0usize;
                let mut emb: *const f32 = ptr::null();
                let mut en = 0usize;
                let mut pose: *const f32 = ptr::null();
                let mut pn = 0usize;
                check_status(unsafe { ffi::md_result_insightface_kps_batch(self.handle, img, i, &mut kps, &mut kn) })?;
                check_status(unsafe { ffi::md_result_insightface_embedding_batch(self.handle, img, i, &mut emb, &mut en) })?;
                check_status(unsafe { ffi::md_result_insightface_pose_batch(self.handle, img, i, &mut pose, &mut pn) })?;
                per_img.push(InsightFace {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    score: it.score,
                    keypoints: slice_items(kps, kn)
                        .iter()
                        .map(|p| Point { x: p.x, y: p.y })
                        .collect(),
                    embedding: read_f32(emb, en),
                    pose: read_f32(pose, pn),
                    gender: it.gender,
                    age: it.age,
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    pub fn ocr_batch(&self) -> Result<Vec<Vec<OcrLine>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut per_img = Vec::new();
            for line in 0.. {
                let mut quad: *const i32 = ptr::null();
                let mut text: *const libc::c_char = ptr::null();
                let mut score = 0f32;
                let status = unsafe { ffi::md_result_ocr_batch(self.handle, img, line, &mut quad, &mut text, &mut score) };
                if status != ffi::MDStatus::OK {
                    break;
                }
                let mut q = [0i32; 8];
                if !quad.is_null() {
                    let s = unsafe { std::slice::from_raw_parts(quad, 8) };
                    q.copy_from_slice(s);
                }
                let mut cls_label = 0;
                let mut cls_score = 0f32;
                let _ = unsafe { ffi::md_result_ocr_cls_batch(self.handle, img, line, &mut cls_label, &mut cls_score) };
                per_img.push(OcrLine {
                    quad: q,
                    text: unsafe { cstr_to_string(text) },
                    score,
                    cls_label,
                    cls_score,
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    pub fn lpr_batch(&self) -> Result<Vec<Vec<LicensePlate>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDLprItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_lpr_batch(self.handle, img, &mut items, &mut n) })?;
            let items = slice_items(items, n);
            let mut per_img = Vec::with_capacity(n);
            for (i, it) in items.iter().enumerate() {
                let mut plate: *const libc::c_char = ptr::null();
                let mut color: *const libc::c_char = ptr::null();
                let mut kps: *const ffi::MDPoint = ptr::null();
                let mut kn = 0usize;
                check_status(unsafe { ffi::md_result_plate_batch(self.handle, img, i, &mut plate, &mut color) })?;
                check_status(unsafe { ffi::md_result_lpr_keypoints_batch(self.handle, img, i, &mut kps, &mut kn) })?;
                per_img.push(LicensePlate {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    plate: unsafe { cstr_to_string(plate) },
                    color: unsafe { cstr_to_string(color) },
                    score: it.score,
                    keypoints: slice_items(kps, kn)
                        .iter()
                        .map(|p| Point { x: p.x, y: p.y })
                        .collect(),
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    pub fn attribute_batch(&self) -> Result<Vec<Vec<Attribute>>, MdError> {
        let images = self.count()?;
        let mut out = Vec::with_capacity(images);
        for img in 0..images {
            let mut items: *const ffi::MDAttrItem = ptr::null();
            let mut n = 0usize;
            check_status(unsafe { ffi::md_result_attribute_batch(self.handle, img, &mut items, &mut n) })?;
            let items = slice_items(items, n);
            let mut per_img = Vec::with_capacity(n);
            for (i, it) in items.iter().enumerate() {
                let mut scores: *const f32 = ptr::null();
                let mut sn = 0usize;
                check_status(unsafe { ffi::md_result_attr_scores_batch(self.handle, img, i, &mut scores, &mut sn) })?;
                per_img.push(Attribute {
                    rect: Rect { x: it.x, y: it.y, width: it.w, height: it.h },
                    box_label_id: it.box_label_id,
                    box_score: it.box_score,
                    attr_scores: read_f32(scores, sn),
                });
            }
            out.push(per_img);
        }
        Ok(out)
    }

    /// 把结果绘制到图像上（句柄直达 C++ vis_*）
    pub fn draw(&self, image: &Image, options: &DrawOptions) -> Result<(), MdError> {
        let mut label_names: Vec<CString> = options
            .label_map
            .iter()
            .map(|(_, n)| CString::new(n.clone()).unwrap_or_default())
            .collect();
        let labels: Vec<ffi::MDLabelItem> = options
            .label_map
            .iter()
            .zip(label_names.iter())
            .map(|((id, _), cname)| ffi::MDLabelItem {
                id: *id,
                name: cname.as_ptr(),
            })
            .collect();
        let font = CString::new(options.font_path.clone()).unwrap_or_default();
        let native = ffi::MDDrawOptions {
            threshold: options.threshold,
            label_map: if labels.is_empty() { ptr::null() } else { labels.as_ptr() },
            label_map_size: labels.len(),
            font_path: if options.font_path.is_empty() { ptr::null() } else { font.as_ptr() },
            font_size: options.font_size,
            alpha: options.alpha,
            save_result: options.save_result as i32,
        };
        let _ = &mut label_names;
        check_status(unsafe { ffi::md_draw_result(image.handle, self.handle, &native) })
    }
}

impl Drop for RawResult {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_result_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 绘制选项（对应 capi MDDrawOptions）
#[derive(Default)]
pub struct DrawOptions {
    pub threshold: f64,
    pub label_map: Vec<(i32, String)>,
    pub font_path: String,
    pub font_size: i32,
    pub alpha: f64,
    pub save_result: bool,
}

impl DrawOptions {
    pub fn new() -> Self {
        Self {
            threshold: 0.5,
            font_size: 14,
            alpha: 0.15,
            save_result: false,
            ..Default::default()
        }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    pub fn with_label_map(mut self, map: Vec<(i32, String)>) -> Self {
        self.label_map = map;
        self
    }

    pub fn with_font(mut self, path: &str, size: i32) -> Self {
        self.font_path = path.to_string();
        self.font_size = size;
        self
    }

    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }
}

// ════════════════════════════════════════════════════════════════
// 各模型一对一包装（对应 C++ 类，与 C# 的 Model 类一一映射）
// ════════════════════════════════════════════════════════════════

/// 宏：生成"持有 Model"的公共包装骨架（struct + new/clone/is_ready + 各 setter）
macro_rules! model_wrapper_common {
    ($name:ident, $kind:expr) => {
        /// 对应 C++ 类（见文档注释）
        pub struct $name {
            inner: Model,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl $name {
            /// 加载模型
            pub fn new(model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
                Ok(Self {
                    inner: Model::new($kind, model_path, option)?,
                })
            }

            /// 深拷贝（复用 backend session）
            /// 与 std Clone::clone 不同：返回 Result<Self, MdError>（深拷贝可失败）。
            #[allow(clippy::should_implement_trait)]
            pub fn clone(&self) -> Result<Self, MdError> {
                Ok(Self {
                    inner: self.inner.clone()?,
                })
            }

            pub fn is_ready(&self) -> bool {
                self.inner.is_ready()
            }

            /// 设置输入尺寸（pipeline 模型设置检测子模型尺寸）
            pub fn set_input_size(&self, w: i32, h: i32) -> Result<(), MdError> {
                self.inner.set_input_size(w, h)
            }

            /// 设置 pipeline 分类子模型输入尺寸（PedestrianAttribute）
            pub fn set_cls_input_size(&self, w: i32, h: i32) -> Result<(), MdError> {
                self.inner.set_cls_input_size(w, h)
            }

            /// 设置 pipeline 分类子模型 batch 大小（>0 固定，-1 自动；PedestrianAttribute）
            pub fn set_cls_batch_size(&self, batch: i32) -> Result<(), MdError> {
                self.inner.set_cls_batch_size(batch)
            }

            /// 设置 OCR 识别子模型 batch 大小（>0 固定，-1 自动）
            pub fn set_rec_batch_size(&self, batch: i32) -> Result<(), MdError> {
                self.inner.set_rec_batch_size(batch)
            }

            /// 设置 OCR 识别子模型输入形状 (c, h, w)
            pub fn set_rec_image_shape(&self, c: i32, h: i32, w: i32) -> Result<(), MdError> {
                self.inner.set_rec_image_shape(c, h, w)
            }

            /// 检测置信度阈值（detection/obb/pose/iseg/facedet/lpr_det 等）
            pub fn set_conf_threshold(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_conf_threshold(v)
            }

            /// NMS 阈值
            pub fn set_nms_threshold(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_nms_threshold(v)
            }

            /// 姿态：关键点数量
            pub fn set_keypoints_num(&self, v: i64) -> Result<(), MdError> {
                self.inner.set_keypoints_num(v)
            }

            /// 实例分割：掩码阈值
            pub fn set_mask_threshold(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_mask_threshold(v)
            }

            /// 分类：Top-K
            pub fn set_top_k(&self, v: i64) -> Result<(), MdError> {
                self.inner.set_top_k(v)
            }

            /// 分类：多标签
            pub fn set_multi_label(&self, v: bool) -> Result<(), MdError> {
                self.inner.set_multi_label(v)
            }

            /// 人脸检测：每人脸关键点数量
            pub fn set_landmarks_per_face(&self, v: i64) -> Result<(), MdError> {
                self.inner.set_landmarks_per_face(v)
            }

            /// OCR det：DB 二值化阈值
            pub fn set_det_db_thresh(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_det_db_thresh(v)
            }

            /// OCR det：DB 框阈值
            pub fn set_det_db_box_thresh(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_det_db_box_thresh(v)
            }

            /// OCR det：unclip 比率
            pub fn set_det_db_unclip_ratio(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_det_db_unclip_ratio(v)
            }

            /// OCR det：得分模式（字符串）
            pub fn set_det_db_score_mode(&self, v: &str) -> Result<(), MdError> {
                self.inner.set_det_db_score_mode(v)
            }

            /// OCR det：膨胀
            pub fn set_use_dilation(&self, v: bool) -> Result<(), MdError> {
                self.inner.set_use_dilation(v)
            }

            /// OCR cls：方向分类阈值
            pub fn set_cls_thresh(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_cls_thresh(v)
            }

            /// 行人属性：检测阈值
            pub fn set_det_threshold(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_det_threshold(v)
            }

            /// insightface：检测阈值
            pub fn set_det_thresh(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_det_thresh(v)
            }

            /// OCR det：最长边（缩放主控）
            pub fn set_max_side_len(&self, v: i64) -> Result<(), MdError> {
                self.inner.set_max_side_len(v)
            }

            /// 车牌检测：每车牌关键点数量
            pub fn set_landmarks_per_card(&self, v: f64) -> Result<(), MdError> {
                self.inner.set_landmarks_per_card(v)
            }
        }
    };
}

/// 宏：生成一个"持有 Model + 类型化 predict(返回 Vec<Item>) + predict_batch(返回 Vec<Vec<Item>>，按图)"的模型包装
macro_rules! model_wrapper {
    ($name:ident, $kind:expr, $reader:expr, $reader_batch:expr) => {
        model_wrapper_common!($name, $kind);

        impl $name {
            /// 推理并读取结果（结果句柄随返回值释放）
            pub fn predict(&self, image: &Image) -> Result<Vec<<$name as ResultType>::Item>, MdError> {
                $reader(&self.inner.predict(image)?)
            }

            /// 批量推理并读取结果（2D：按图返回，每图一组，保留图片边界）
            pub fn predict_batch(&self, images: &[&Image]) -> Result<Vec<Vec<<$name as ResultType>::Item>>, MdError> {
                $reader_batch(&self.inner.predict_batch(images)?)
            }

            /// 推理并把结果绘制到图像上（句柄直达 C++ vis_*）
            pub fn predict_and_draw(
                &self,
                image: &Image,
                canvas: &Image,
                options: &DrawOptions,
            ) -> Result<Vec<<$name as ResultType>::Item>, MdError> {
                let result = self.inner.predict(image)?;
                result.draw(canvas, options)?;
                $reader(&result)
            }
        }
    };
}

/// 宏：生成一个标量模型包装（predict 直接返回 `Item`，无 predict_batch）
macro_rules! scalar_model_wrapper {
    ($name:ident, $kind:expr, $reader:expr) => {
        model_wrapper_common!($name, $kind);

        impl $name {
            /// 推理并读取标量结果（结果句柄随返回值释放）
            pub fn predict(&self, image: &Image) -> Result<<$name as ResultType>::Item, MdError> {
                $reader(&self.inner.predict(image)?)
            }
        }
    };
}

/// 辅助 trait：每个模型的 predict 返回类型
pub trait ResultType {
    type Item;
}

impl ResultType for UltralyticsDet {
    type Item = Detection;
}
impl ResultType for Classification {
    type Item = ClassificationResult;
}
impl ResultType for UltralyticsPose {
    type Item = Pose;
}
impl ResultType for HandKeypoint {
    type Item = Pose;
}
impl ResultType for UltralyticsObb {
    type Item = Obb;
}
impl ResultType for UltralyticsSeg {
    type Item = InstanceSeg;
}
impl ResultType for UltralyticsSem {
    type Item = SemSeg;
}
impl ResultType for UltralyticsDepth {
    type Item = Depth;
}
impl ResultType for Scrfd {
    type Item = FaceDetection;
}
impl ResultType for SeetaFaceID {
    type Item = FaceRecognition;
}
impl ResultType for SeetaFaceAge {
    type Item = i32;
}
impl ResultType for SeetaFaceGender {
    type Item = i32;
}
impl ResultType for InsightFaceAnalysis {
    type Item = InsightFace;
}
impl ResultType for PaddleOCR {
    type Item = OcrLine;
}
impl ResultType for LprPipeline {
    type Item = LicensePlate;
}
impl ResultType for PedestrianAttribute {
    type Item = Attribute;
}
impl ResultType for ReID {
    type Item = ReIdResult;
}

model_wrapper!(UltralyticsDet, ModelKind::Detection, RawResult::detection, RawResult::detection_batch);
model_wrapper!(Classification, ModelKind::Classification, RawResult::classification, RawResult::classification_batch);
model_wrapper!(UltralyticsPose, ModelKind::Pose, RawResult::pose, RawResult::pose_batch);
model_wrapper!(HandKeypoint, ModelKind::Hand, RawResult::pose, RawResult::pose_batch);
model_wrapper!(UltralyticsObb, ModelKind::Obb, RawResult::obb, RawResult::obb_batch);
model_wrapper!(UltralyticsSeg, ModelKind::InstanceSeg, RawResult::instance_seg, RawResult::instance_seg_batch);
model_wrapper!(UltralyticsSem, ModelKind::SemSeg, |r: &RawResult| r.sem_seg().map(|s| vec![s]), |r: &RawResult| r.sem_seg_batch());
model_wrapper!(UltralyticsDepth, ModelKind::Depth, |r: &RawResult| r.depth().map(|d| vec![d]), |r: &RawResult| r.depth_batch());
model_wrapper!(Scrfd, ModelKind::FaceDet, RawResult::face_det, RawResult::face_det_batch);
model_wrapper!(SeetaFaceID, ModelKind::FaceRec, |r: &RawResult| r.face_recognition(0).map(|f| vec![f]), |r: &RawResult| r.face_recognition_batch());
scalar_model_wrapper!(SeetaFaceAge, ModelKind::FaceAge, RawResult::age);
scalar_model_wrapper!(SeetaFaceGender, ModelKind::FaceGender, RawResult::gender);
model_wrapper!(InsightFaceAnalysis, ModelKind::InsightFace, RawResult::insightface, RawResult::insightface_batch);
model_wrapper!(PaddleOCR, ModelKind::Ocr, RawResult::ocr, RawResult::ocr_batch);
model_wrapper!(LprPipeline, ModelKind::LprPipeline, RawResult::lpr, RawResult::lpr_batch);
model_wrapper!(PedestrianAttribute, ModelKind::PedestrianAttribute, RawResult::attribute, RawResult::attribute_batch);
model_wrapper!(ReID, ModelKind::ReId, |r: &RawResult| r.reid(0).map(|x| vec![x]), RawResult::reid_batch);

// ═══ 子模型（OCR / LPR / insightface 组件，可独立部署） ═══

impl ResultType for DbDetectorModel {
    type Item = OcrLine;
}
impl ResultType for RecognizerModel {
    type Item = OcrLine;
}
impl ResultType for OcrClassifierModel {
    type Item = OcrLine;
}
impl ResultType for LprDetectionModel {
    type Item = LicensePlate;
}
impl ResultType for LprRecognizerModel {
    type Item = LicensePlate;
}
impl ResultType for InsightFaceDetModel {
    type Item = FaceDetection;
}
impl ResultType for FaceRecognizerPipelineModel {
    type Item = FaceRecognition;
}

model_wrapper!(DbDetectorModel, ModelKind::OcrDet, RawResult::ocr, RawResult::ocr_batch);
model_wrapper!(RecognizerModel, ModelKind::OcrRec, RawResult::ocr, RawResult::ocr_batch);
model_wrapper!(OcrClassifierModel, ModelKind::OcrCls, RawResult::ocr, RawResult::ocr_batch);
model_wrapper!(LprDetectionModel, ModelKind::LprDet, RawResult::lpr, RawResult::lpr_batch);
model_wrapper!(LprRecognizerModel, ModelKind::LprRec, RawResult::lpr, RawResult::lpr_batch);
model_wrapper!(InsightFaceDetModel, ModelKind::InsightFaceDet, RawResult::face_det, RawResult::face_det_batch);
model_wrapper!(FaceRecognizerPipelineModel, ModelKind::FaceRecPipeline, RawResult::face_recognition_all, RawResult::face_recognition_batch);

// ═══ 音频模型（非 predict 形态，单独实现） ═══

/// 对应 C++ SenseVoice（ASR）
pub struct SenseVoice {
    inner: Model,
}

unsafe impl Send for SenseVoice {}
unsafe impl Sync for SenseVoice {}

impl SenseVoice {
    /// 路径格式: model.onnx|tokens.txt
    pub fn new(model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        Ok(Self {
            inner: Model::new(ModelKind::Asr, model_path, option)?,
        })
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, MdError> {
        Ok(Self {
            inner: self.inner.clone()?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// 从 wav 文件识别文本
    pub fn predict_wav(&self, wav_path: &str) -> Result<String, MdError> {
        self.inner.asr_wav(wav_path)
    }

    /// 从 PCM float 采样识别文本
    pub fn predict(&self, samples: &[f32], sample_rate: i32) -> Result<String, MdError> {
        self.inner.asr(samples, sample_rate)
    }
}

/// 对应 C++ Kokoro（TTS）
pub struct Kokoro {
    inner: Model,
}

unsafe impl Send for Kokoro {}
unsafe impl Sync for Kokoro {}

impl Kokoro {
    /// 路径格式: model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir
    pub fn new(model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        Ok(Self {
            inner: Model::new(ModelKind::Tts, model_path, option)?,
        })
    }

    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Result<Self, MdError> {
        Ok(Self {
            inner: self.inner.clone()?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// 文本合成音频
    pub fn predict(&self, text: &str, voice: &str, speed: f32) -> Result<TtsAudio, MdError> {
        self.inner.tts(text, voice, speed)
    }
}

fn slice_items<T>(ptr: *const T, n: usize) -> &'static [T] {
    if n == 0 || ptr.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }
}

fn read_bytes(ptr: *const u8, n: usize) -> Vec<u8> {
    if n == 0 || ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }
}

fn read_f32(ptr: *const f32, n: usize) -> Vec<f32> {
    if n == 0 || ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }
}

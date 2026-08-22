use crate::error::{check_status, MdError};
use crate::ffi;
use crate::types::*;
use std::ffi::CString;
use std::ptr;

/// 多目标跟踪器（capi 单一分发：ByteTrack / BoT-SORT / StrongSORT）。
/// 纯 CPU 无模型依赖；跟踪 ID 跨帧稳定，reset() 归零。
pub struct Tracker {
    pub(crate) handle: ffi::MDTrackerHandle,
    pub(crate) kind: TrackerKind,
}

unsafe impl Send for Tracker {}
unsafe impl Sync for Tracker {}

impl Tracker {
    /// 按类型创建跟踪器。
    pub fn new(kind: TrackerKind) -> Result<Self, MdError> {
        let mut handle = ptr::null_mut();
        check_status(unsafe { ffi::md_tracker_create(kind.to_ffi(), &mut handle) })?;
        if handle.is_null() {
            return Err(MdError::ModelInit("tracker".into()));
        }
        Ok(Self { handle, kind })
    }

    /// 跟踪器类型。
    pub fn kind(&self) -> TrackerKind {
        self.kind
    }

    /// 逐帧更新：输入检测框，返回跟踪目标（ID 跨帧稳定）。
    ///
    /// 内部采用两阶段容量探测：先以容量 0 探测所需大小，再分配后正式写入，
    /// 并对写入数做边界约束，避免读取越界。
    pub fn update(
        &mut self,
        boxes: &[Rect],
        scores: &[f32],
        label_ids: &[i32],
    ) -> Result<Vec<TrackItem>, MdError> {
        let n = boxes.len();
        if scores.len() < n || label_ids.len() < n {
            return Err(MdError::InvalidArgument(
                "boxes/scores/label_ids length mismatch".into(),
            ));
        }

        let boxes_ffi: Vec<ffi::MDBox> = boxes
            .iter()
            .map(|r| ffi::MDBox {
                x: r.x,
                y: r.y,
                w: r.width,
                h: r.height,
            })
            .collect();

        // 阶段 1：容量探测。capi 要求 out 非空，故用 dangling（非空）指针，
        // 容量 0 下必然不足，返回 ERR_INVALID_ARGUMENT 并在 out_count 写入所需数。
        let mut probe_cap: usize = 0;
        let probe = ptr::NonNull::<ffi::MDTrackItem>::dangling();
        let status = unsafe {
            ffi::md_tracker_update(
                self.handle,
                boxes_ffi.as_ptr(),
                scores.as_ptr(),
                label_ids.as_ptr(),
                n,
                probe.as_ptr(),
                &mut probe_cap,
            )
        };
        if status == ffi::MDStatus::OK {
            // 本帧无跟踪目标，容量保持 0
            return Ok(Vec::new());
        }
        if status != ffi::MDStatus::ERR_INVALID_ARGUMENT {
            return Err(status_err(status));
        }
        let need = probe_cap;

        // 阶段 2：分配 need 个目标，正式写入
        let mut out = vec![
            ffi::MDTrackItem {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
                track_id: 0,
                label_id: 0,
                score: 0.0,
                state: 0,
            };
            need
        ];
        let mut written = need;
        check_status(unsafe {
            ffi::md_tracker_update(
                self.handle,
                boxes_ffi.as_ptr(),
                scores.as_ptr(),
                label_ids.as_ptr(),
                n,
                out.as_mut_ptr(),
                &mut written,
            )
        })?;
        let written = written.min(out.len());

        Ok(out[..written]
            .iter()
            .map(|it| TrackItem {
                rect: Rect {
                    x: it.x,
                    y: it.y,
                    width: it.w,
                    height: it.h,
                },
                track_id: it.track_id,
                label_id: it.label_id,
                score: it.score,
                state: TrackState::try_from(it.state).unwrap_or(TrackState::Removed),
            })
            .collect())
    }

    /// 通用命名参数设置（double 值）。
    pub fn set_param(&self, name: &str, value: f64) -> Result<(), MdError> {
        let cn = CString::new(name).map_err(|_| MdError::InvalidArgument("name".into()))?;
        check_status(unsafe { ffi::md_tracker_set_params(self.handle, cn.as_ptr(), value) })
    }

    /// 重置跟踪器（ID 归零）。
    pub fn reset(&self) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_tracker_reset(self.handle) })
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::md_tracker_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// 将非 OK 状态码转为 MdError（附最近错误信息）。
fn status_err(code: ffi::MDStatus) -> MdError {
    let err = unsafe { ffi::md_get_last_error() };
    let last = if err.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned() }
    };
    MdError::from_status(code, &last)
}

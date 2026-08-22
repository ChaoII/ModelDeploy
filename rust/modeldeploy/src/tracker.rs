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
    /// 遵循 capi 的查询/提交契约：先以 md_tracker_capacity 做**非变异**容量查询（不推进
    /// 跟踪器状态），分配所需缓冲后再以恰好的容量调用有状态 md_tracker_update **一次**。
    /// 这样每逻辑帧跟踪器只推进一次（Kalman / max_age / ID 计数正确），避免旧的两阶段
    /// 探测（用 update 自身探测）导致的双重推进。空帧（无输出）也会照常调用 update 推进状态。
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

        // 阶段 1：非变异容量查询（不推进状态）
        let mut need: usize = 0;
        check_status(unsafe {
            ffi::md_tracker_capacity(
                self.handle,
                boxes_ffi.as_ptr(),
                scores.as_ptr(),
                label_ids.as_ptr(),
                n,
                &mut need,
            )
        })?;

        // 阶段 2：分配 need 个目标，正式写入（有状态，恰一次）。
        // 空帧（need==0）也要调用 update 推进状态；分配 max(need,1) 保证传出指针非空，
        // 满足 capi 的 out 空指针守卫（不写越界，写 0 项）。
        let alloc = if need == 0 { 1 } else { need };
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
            alloc
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

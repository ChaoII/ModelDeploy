use crate::error::{check_status, MdError};
use crate::ffi;
use std::ptr;

/// 跨线目标计数（capi md_solution_object_counter_*）。
pub struct ObjectCounter {
    handle: ffi::MDSolutionHandle,
}

unsafe impl Send for ObjectCounter {}
unsafe impl Sync for ObjectCounter {}

impl ObjectCounter {
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_solution_create(&mut h, ffi::MDSolutionKind::ObjectCounter as i32) })?;
        Ok(Self { handle: h })
    }

    pub fn set_line(&self, a: (f32, f32), b: (f32, f32)) -> Result<(), MdError> {
        check_status(unsafe {
            ffi::md_solution_object_counter_set_line(self.handle, a.0, a.1, b.0, b.1)
        })
    }

    pub fn update(&self, boxes: &[f32], labels: &[i32], track_ids: &[i32]) -> Result<(), MdError> {
        check_status(unsafe {
            ffi::md_solution_object_counter_update(
                self.handle,
                boxes.as_ptr(),
                labels.len(),
                labels.as_ptr(),
                track_ids.as_ptr(),
            )
        })
    }

    pub fn hline(&self) -> Result<(i32, i32), MdError> {
        let (mut i, mut o) = (0i32, 0i32);
        check_status(unsafe { ffi::md_solution_object_counter_hline(self.handle, &mut i, &mut o) })?;
        Ok((i, o))
    }
}

impl Drop for ObjectCounter {
    fn drop(&mut self) {
        unsafe { ffi::md_solution_destroy(self.handle) };
    }
}

/// 轨迹热力图（capi md_solution_heatmap_*）。
pub struct Heatmap {
    handle: ffi::MDSolutionHandle,
}

unsafe impl Send for Heatmap {}
unsafe impl Sync for Heatmap {}

impl Heatmap {
    pub fn new() -> Result<Self, MdError> {
        let mut h = ptr::null_mut();
        check_status(unsafe { ffi::md_solution_create(&mut h, ffi::MDSolutionKind::Heatmap as i32) })?;
        Ok(Self { handle: h })
    }

    pub fn set_size(&self, w: i32, h: i32) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_solution_heatmap_set_size(self.handle, w, h) })
    }

    pub fn update(&self, boxes: &[f32], frame_w: i32, frame_h: i32) -> Result<(), MdError> {
        check_status(unsafe {
            ffi::md_solution_heatmap_update(self.handle, boxes.as_ptr(), boxes.len() / 4, frame_w, frame_h)
        })
    }

    pub fn peak(&self) -> Result<(i32, i32), MdError> {
        let (mut x, mut y) = (0i32, 0i32);
        check_status(unsafe { ffi::md_solution_heatmap_peak(self.handle, &mut x, &mut y) })?;
        Ok((x, y))
    }
}

impl Drop for Heatmap {
    fn drop(&mut self) {
        unsafe { ffi::md_solution_destroy(self.handle) };
    }
}

/// 纯工具：两个矩形（x,y,w,h）的交并比。8 个坐标参数语义独立，属不可避免的参数数量。
#[allow(clippy::too_many_arguments)]
pub fn iou(ax: f32, ay: f32, aw: f32, ah: f32, bx: f32, by: f32, bw: f32, bh: f32) -> Result<f32, MdError> {
    let mut out = 0.0f32;
    check_status(unsafe { ffi::md_vision_iou4(ax, ay, aw, ah, bx, by, bw, bh, &mut out) })?;
    Ok(out)
}

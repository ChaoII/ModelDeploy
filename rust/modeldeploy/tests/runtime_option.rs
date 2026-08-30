use modeldeploy::ffi::MDDevice;
use modeldeploy::{MdError, RuntimeOption};

// 不初始化后端，仅验证 RuntimeOption setter 接线；即使 build_win 为 ORT-only 也不会触发 MD_LOG_FATAL。

#[test]
fn set_device_and_config_ok() -> Result<(), MdError> {
    let mut opt = RuntimeOption::new()?;
    opt.set_device(modeldeploy::ffi::MDDevice::GPU, 1)?;
    opt.set_config("ort", "trt_engine_cache_path", ".")?;
    Ok(())
}

#[test]
fn set_device_opencl_vulkan_ok() -> Result<(), MdError> {
    let mut opt = RuntimeOption::new()?;
    opt.use_mnn().set_device(MDDevice::OPENCL, 0)?;
    opt.set_device(MDDevice::VULKAN, 0)?;
    Ok(())
}

#[test]
fn unknown_config_namespace_err() -> Result<(), MdError> {
    let mut opt = RuntimeOption::new()?;
    let r = opt.set_config("bogus", "k", "v");
    assert!(r.is_err(), "未知 config key 应返回 Err");
    Ok(())
}

#[test]
fn validate_ptr_cpu() {
    let x = 5u8;
    let ok = RuntimeOption::new()
        .unwrap()
        .validate_ptr_device(&x as *const u8 as *const std::ffi::c_void, MDDevice::CPU, 0)
        .unwrap();
    assert!(ok);
    let ok2 = RuntimeOption::new()
        .unwrap()
        .validate_ptr_device(std::ptr::null(), MDDevice::CPU, 0)
        .unwrap();
    assert!(!ok2);
}

#[test]
fn validate_ptr_tpu_unsupported() {
    let x = 5u8;
    let r = RuntimeOption::new()
        .unwrap()
        .validate_ptr_device(&x as *const u8 as *const std::ffi::c_void, MDDevice::TPU, 0);
    assert!(r.is_err());
}

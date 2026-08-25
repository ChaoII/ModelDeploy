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
fn unknown_config_namespace_err() -> Result<(), MdError> {
    let mut opt = RuntimeOption::new()?;
    let r = opt.set_config("bogus", "k", "v");
    assert!(r.is_err(), "未知 config key 应返回 Err");
    Ok(())
}

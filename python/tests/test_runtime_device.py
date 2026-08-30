"""RuntimeOption 设备 setter 接线测试（不依赖模型/数据，不触发真实后端初始化）。

对应 C/C C#/Rust 的 OPENCL/VULKAN set_device 测试，验证五语言暴露面语义对齐：
OPENCL/VULKAN 需显式 use_mnn_backend()，否则 fail-closed。
"""

import modeldeploy as md


def test_set_device_opencl_vulkan():
    opt = md.RuntimeOption()
    opt.use_mnn_backend()
    opt.set_device(md.Device.OPENCL, 0)
    opt.set_device(md.Device.VULKAN, 0)


def test_set_device_default_device_id():
    opt = md.RuntimeOption()
    opt.use_mnn_backend()
    opt.set_device(md.Device.VULKAN)

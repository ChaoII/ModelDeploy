import pathlib
import sys

PKG_ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, PKG_ROOT)

import modeldeploy
from modeldeploy import RuntimeOption
import modeldeploy.vision as vision


def test_runtime_option_path():
    opt = RuntimeOption()
    opt.set_model_path(pathlib.Path("dummy.onnx"))
    # set_model_path 只存字符串，验证不再 TypeError 即可
    print("runtime_option_path OK")


def test_vision_ctor_path():
    # 构造器接受 Path（不真正加载，模型不存在也能走到类型转换后的构造入口）
    opt = RuntimeOption()
    opt.use_cpu()
    try:
        m = vision.UltralyticsDet(pathlib.Path("nonexistent.onnx"), opt)
        print("det_ctor_path OK (loaded),", type(m).__name__)
    except Exception as e:
        # 若因找不到模型文件而抛异常，说明类型转换已通过（非 TypeError）
        assert "Path" not in str(type(e).__name__) and not isinstance(e, TypeError), e
        print("det_ctor_path OK (reached ctor, err =", type(e).__name__, ")")


if __name__ == "__main__":
    test_runtime_option_path()
    test_vision_ctor_path()
    print("ALL SMOKE OK")

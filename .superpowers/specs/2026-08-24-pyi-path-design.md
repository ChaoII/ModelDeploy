# 设计：pyi 构建期自动生成 + pybind 路径形参支持 pathlib.Path

日期：2026-08-24
状态：已获用户确认

## 背景与目标

1. **pyi 接口文件**：当前 `.pyi` 是手动运行 `pybind11-stubgen modeldeploy` 得到的并提交（仅 `__init__/vision/audio` 三个，覆盖不全）。`python -m build` 时**不会**重新生成。目标：构建期自动生成、覆盖全部子模块、随 wheel 一起发布。
2. **路径入参**：绑定的路径形参全部是 `std::string`，Python 端传 `pathlib.Path` 会失败。目标：所有路径类形参同时接受 `str`/`bytes`/`os.PathLike`。

## 现状调查结论

- pybind 目录 55 个 `.cpp`、约 510 处 `.def`；路径类 `std::string` 形参约 **58 个函数 / 80 处**，主要集中在模型构造器 `init<std::string, RuntimeOption>` 中。
- 公共头 `csrc/pybind/utils/utils.h` 已被 40 个绑定文件 include，是放置自定义 type_caster 的最佳位置。
- C++ 内核 API 全部使用 `std::string` 表示路径（0 处 `std::filesystem::path`）。
- pybind11 不能安全全局覆盖内置 `std::string` caster，故 Path 兼容需在绑定层改造形参类型。

## 第 1 条线：pyi 构建期自动生成

### 依赖
- `pyproject.toml` → `[build-system].requires` 增加 `pybind11-stubgen`、`numpy`（模块 import 需要 numpy）。

### 机制
- 在根 `CMakeLists.txt` 的 `BUILD_PYTHON` 块、所有 `install()` 之后，增加一个 `install(CODE ...)`：
  - install 阶段（scikit-build-core 组装 wheel 的 staging）用当前 `Python_EXECUTABLE` 执行
    `python -m pybind11_stubgen <modeldeploy> --output-dir <staging>`，
    并设 `PYTHONPATH` 指向 staging（此时 pyd + DLL 同目录，可成功 import）。
  - 输出写回 staging 的 `modeldeploy/`，scikit-build-core 将生成的 pyi 打进 wheel。
  - stubgen 失败记为 warning 而非 build 失败（用 `execute_process(... RESULT_VARIABLE)` 判断），保证罕见类型导致 stub 失败时 wheel 仍能产出（仅无 pyi）。

### 结果
每次 `python -m build` 的 wheel 都带最新、覆盖全部子模块（含 pipeline/video/nlp/runtime）的 pyi。

## 第 2 条线：路径形参支持 pathlib.Path

### 公共 caster
- 在 `csrc/pybind/utils/utils.h` 增加 `pybind11::detail::type_caster<std::filesystem::path>`：
  - `load`：用 `PyOS_FSPath` 同时接受 `str`/`bytes`/任意 `os.PathLike`（pathlib.Path）。
  - caster 名设置为 `os.PathLike[str]`，使生成 pyi 标注为 Path 兼容。

### 机械改造
- 把约 58 个函数 / 80 处路径形参从 `std::string` 改为 `std::filesystem::path`：
  - `init<std::string, RuntimeOption>` 构造 → `init<std::filesystem::path, RuntimeOption>`，用 lambda 内部 `.string()` 转发给 C++ 构造。
  - `set_model_path`、`create_from_dir`、`visual` 的 `font_path` 等成员指针绑定 → 换成 lambda + `.string()`。
- C++ 内核 API 保持不变（仍为 `std::string`），仅在绑定层做 Path→string 网关。

## 测试

- Python 冒烟测试：
  - `RuntimeOption().set_model_path(pathlib.Path(...))` 不抛 TypeError。
  - 典型模型构造器传 `pathlib.Path`（如 `UltralyticsDet(Path(...), runtime_option)`）。
  - `create_from_dir(Path(...))`。
  - 验证生成的 pyi 中相关形参类型为 `os.PathLike[str]`。

## 非目标 / 取舍

- 不修改 SDK 内部 C++ API（`std::string` 保留）。
- stubgen 失败不阻断构建（降级为 warning）。
- 不引入 Python 层包装类来兼容 Path（避免二次开销）。

## 风险与缓解
- stubgen CLI 输出路径/命名需实测确认（`__init__.pyi` vs 单文件）。→ 实现计划中先做一次手工验证再固化命令。
- 大量机械形参改动易遗漏 → 以探索调查清单为核对基准，改完靠编译 + 冒烟测试兜底。

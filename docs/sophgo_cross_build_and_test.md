# Sophgo 交叉编译 / 上传 / 设备测试 SOP

本 SOP 记录 ModelDeploy 在算能(Sophgo) TPU 上「交叉编译 → 传到设备 → 运行验证」的整套流程。当用户说 **“在 sophgo 上测试”**（或类似触发词）时，按本文件执行。

## 0. 核心结论（先读）

- **纯 Sophgo 构建用 `ENABLE_ORT=OFF`**：SDK 与 demo 只依赖 `/opt/sophon` 的运行时库（`libbmrt/bmcv/bmlib/bmjpeg`），**零 onnxruntime 依赖**。
  - onnxruntime 是静态库，本来就不该以 `.so` 动态依赖；且 Sophgo 推理走 Sophgo 后端，根本不用 ORT。
  - 若误用 `ENABLE_ORT=ON`，SDK 会把 onnxruntime 编成 `libonnxruntime.so` 并链入，导致设备端报 “libonnxruntime.so not found”。**切勿在 Sophgo 构建里开 ORT**。
- Sophgo 的 int8 `.bmodel` 多为 **batch=1 静态形状**：多人属性 pipeline 默认 `set_cls_batch_size(8)` 会把裁剪图拼成 batch>1，与后端不匹配 → `std::bad_alloc/bad_array_new_length`。**必须 `set_cls_batch_size(1)`**。
- 子模型输入尺寸（det/cls 的 宽×高）**从 `get_input_info(0).shape` 读取**，不要硬编码（各 bmodel 输入不同）。

## 1. 环境速查

| 项 | 值 |
|----|----|
| 交叉编译宿主机 | `172.168.100.243`（macOS arm64），用户 `aichao`，密码 `1234` |
| Docker | Docker Desktop，CLI 在 `/Applications/Docker.app/Contents/Resources/bin/docker`（需加 PATH） |
| 镜像 / 容器 | 镜像 `tpuc_dev:arm64-sophgo-0414`；运行容器名 `tpuc_dev`（ssh 22→宿主 2222），`docker exec tpuc_dev bash -lc "..."` 以 root 进 |
| 容器内工作区 | `/workspace` = ModelDeploy 仓库 + `build_sophgo`（已存在，复用缓存） |
| 容器内 Sophgo 依赖 | `/opt/sophon`（`libsophon-current`，含 bmrt/bmcv/bmlib/bmjpeg）；OpenCV/其他依赖已 fetch 到 `build_sophgo/_deps` |
| 容器内工具链 | Linux aarch64 (Ubuntu 22.04)，`gcc 11.4`，`cmake 4.4.2` |
| 目标设备 | `172.168.100.70`（`Linux sophon` aarch64，算能板卡），用户 `linaro`，密码 `linaro` |
| 设备运行目录 | `/data/ModelDeploy/build_sophgo/bin`（SDK 库 `libModelDeploySDK.so*` + 各 demo 可执行） |
| 设备 Sophgo 运行时 | `/opt/sophon/libsophon-current/lib`（bmrt/bmcv/bmlib/bmjpeg） |
| 传输链路 | 本机 → `.243` 宿主 → `docker cp` 进容器；产物 `docker cp` 出容器 → 存 `.243` → 直传 `.70` |

本地仓库根：`E:\CLionProjects\ModelDeploy`（Windows）。

## 2. 依赖与前置

SSH/SFTP 非交互式密码登录：用 Python `paramiko`（本机 `python -c "import paramiko"` 已装 5.x）。复用以下最小工具逻辑：

```python
def connect(host, user, pwd, port=22):
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pwd, timeout=15,
              allow_agent=False, look_for_keys=False)
    return c

def run(c, cmd, timeout=300):
    _, out, err = c.exec_command(cmd, timeout=timeout, get_pty=False)
    rc = out.channel.recv_exit_status()
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    if o: print(o.rstrip())
    if e: print("[stderr]", e.rstrip())
    return rc
```

在 `.243` 的 zsh 里，`echo ===x===` 会被 `=x` 展开坑到，务必给 echo 参数加引号或用 `echo "---x---"`。

## 3. 标准流程

### 3.1 上传源码/模型到容器

把需要更新的文件从本机经 `.243` 传给 `tpuc_dev:/workspace/...`。**注意用正斜杠拼 SFTP 远端路径**（`os.path.join` 在 Windows 会用反斜杠，导致容器里出现 `md_stage\xxx` 这样的脏文件名）。

```python
HOST, USER, PWD, CONTAINER = "172.168.100.243", "aichao", "1234", "tpuc_dev"
FILES = [  # (本地绝对路径, 容器内 /workspace 相对路径)
    (r"E:\CLionProjects\ModelDeploy\examples\demo_pipeline\demo_pedestrian_attribute_sophgo.cpp",
     "examples/demo_pipeline/demo_pedestrian_attribute_sophgo.cpp"),
    (r"...\CMakeLists.txt", "examples/demo_pipeline/CMakeLists.txt"),
    (r"E:\...\zhgd_without_nms_640_int8.bmodel", "test_data/test_models/sophgo/zhgd_without_nms_640_int8.bmodel"),
    (r"E:\...\zhgd_ml_int8.bmodel",              "test_data/test_models/sophgo/zhgd_ml_int8.bmodel"),
    (r"E:\...\test_pedestrian_attribute_scale.png","test_data/test_images/test_pedestrian_attribute_scale.png"),
]
c = connect(HOST, USER, PWD)
stage = "/Users/aichao/md_stage"
run(c, "rm -rf " + stage + "; mkdir -p " + stage)
sftp = c.open_sftp()
for local, rel in FILES:
    sftp.put(local, stage + "/" + os.path.basename(local))   # 正斜杠
sftp.close()
run(c, f'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"; '
       f'docker exec {CONTAINER} bash -lc "mkdir -p '
       f'/workspace/examples/demo_pipeline /workspace/test_data/test_models/sophgo /workspace/test_data/test_images"')
for local, rel in FILES:
    name = os.path.basename(local)
    dparent = "/workspace/" + "/".join(rel.split("/")[:-1])
    run(c, f'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"; '
           f'docker cp "{stage}/{name}" {CONTAINER}:"{dparent}/"')
# 校验容器内已就位
```

> `docker cp` 必须在 **`.243` 宿主**层执行（不要放进 `docker exec ... bash -lc "..."`，容器里没有 docker）。

### 3.2 构建（容器内，纯 Sophgo）

每次改源码后需先重新 configure 以注册新 target，再增量构建目标：

```bash
cd /workspace/build_sophgo \
  && cmake -DENABLE_SOPHGO=ON -DENABLE_ORT=OFF -DENABLE_MNN=OFF -DENABLE_TRT=OFF . \
  && cmake --build . --target demo_pedestrian_attribute_sophgo -j"$(nproc)"
```

关键选项（其余沿用已有 `CMakeCache.txt`）：`ENABLE_SOPHGO=ON`、**`ENABLE_ORT=OFF`**、`ENABLE_MNN/OFF`、`ENABLE_TRT/OFF`、`WITH_GPU=OFF`、`BUILD_VISION=ON`、`BUILD_CAPI=ON`、`BUILD_EXAMPLES=ON`、`CMAKE_BUILD_TYPE=Release`。
构建成功后产物：`/workspace/build_sophgo/bin/{demo_xxx_sophgo, libModelDeploySDK.so.1.8.0}`。
用 `ldd` 确认 demo 只依赖 sophgo 库、无 onnxruntime：
```bash
ldd bin/demo_xxx_sophgo | grep -iE "onnx|bm|not found"
# 期望：只有 libModelDeploySDK.so.1 和 libbmrt/libbmcv/libbmlib/libbmjpeg
```

### 3.3 取产物出容器 → 传设备

在 `.243` 宿主：
```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
mkdir -p /Users/aichao/md_out
docker cp tpuc_dev:/workspace/build_sophgo/bin/demo_pedestrian_attribute_sophgo /Users/aichao/md_out/
docker cp tpuc_dev:/workspace/build_sophgo/bin/libModelDeploySDK.so.1.8.0 /Users/aichao/md_out/
```

`.243 → .70`（用上面 `connect`，读 .243 写 .70）：
```python
BIN = "/data/ModelDeploy/build_sophgo/bin"
for f in ["demo_pedestrian_attribute_sophgo", "libModelDeploySDK.so.1.8.0"]:
    b = s_sftp.open(f"/Users/aichao/md_out/{f}", "rb").read()
    d_sftp.open(f"{BIN}/{f}", "wb").write(b)
# 重建 SDK 软链接（.70 上执行）：
#   ln -sfn libModelDeploySDK.so.1.8.0 libModelDeploySDK.so.1
#   ln -sfn libModelDeploySDK.so.1      libModelDeploySDK.so
#   chmod +x demo_pedestrian_attribute_sophgo
```

依赖模型/测试图已放 `.70`：`/data/ModelDeploy/test_data/...`（与仓库布局一致，保证默认相对路径 `../../test_data/...` 可用）。若多余遗留了 `libonnxruntime.so`，可直接删（纯 Sophgo 构建不需要）：
```bash
rm -f /data/ModelDeploy/build_sophgo/bin/libonnxruntime.so
```

### 3.4 在设备上运行验证

```bash
cd /data/ModelDeploy/build_sophgo/bin
export LD_LIBRARY_PATH=/data/ModelDeploy/build_sophgo/bin:/opt/sophon/libsophon-current/lib
stdbuf -oL -eL ./demo_pedestrian_attribute_sophgo \
    ../../test_data/test_models/sophgo/zhgd_without_nms_640_int8.bmodel \
    ../../test_data/test_models/sophgo/zhgd_ml_int8.bmodel \
    ../../test_data/test_images/test_pedestrian_attribute_scale.png \
    5 0.5
```

`stdbuf -oL -eL` 让 stdout 无缓冲，避免进程崩溃时丢打印（`printf` 到管道是全缓冲）。预期：两个 bmodel 加载、打印输入尺寸、`attributes=N`、属性表、`saved xxx.jpg`。`.70` 无 gdb；`which stdbuf`、`file` 可用做基本诊断。

## 4. 常见坑与排查

| 现象 | 原因 / 处理 |
|------|-------------|
| 报 `libonnxruntime.so not found` | build_sophgo 缓存是 `ENABLE_ORT=ON`。重新 `-DENABLE_ORT=OFF` 纯 Sophgo 构建，删掉遗留 .so |
| `std::bad_alloc` / `bad_array_new_length` | Sophgo int8 bmodel 是 batch=1 静态形状；pipeline 内 `set_cls_batch_size(1)`（默认 8 的 batch>1 与后端不匹配） |
| 分类/检测结果全为 0 或崩溃 | 子模型输入尺寸要从 `get_input_info(0).shape` 读，别硬编码（如 cls `{192,256}`、det `640x640` 各 bmodel 不同） |
| 容器里出现 `md_stage\xxx` 脏文件 | SFTP 远端路径用了 `os.path.join`（Windows 反斜杠）。统一用正斜杠拼路径，并先 `rm` 清理脏文件 |
| `.243` zsh 下 `echo ===x===` 报错 | 被 `=x` 展开坑到；加引号 `echo "---x---"` |
| SSH 偶发 `TimeoutError` | `.243` 网络抖动，稍后重试即可 |
| 检测返回很多零分/零尺寸框 | 可能用了旧版(历史) SDK 二进制；用当前源码重新构建 demo_detection_sophgo 验证 |

## 5. 改造要点（行人属性 Sophgo demo 为例）

- `RuntimeOption`：`option.use_sophgo_backend(0)`；**不设 `sophgo_option.bmodel_path`**，让双子模型各用构造时 `set_model_path` 写入的 `model_file`（Sophgo 后端在 `bmodel_path` 为空时回退 `option.model_file`）——这与单模型 demo 的“直接设 bmodel_path”不同。
- 尺寸自适应：det/cls 都从 `get_detector()/get_classifier()->get_input_info(0).shape` 读取 `{W, H}`。
- `set_cls_batch_size(1)`。
- 输出命名：`vis_attr` 存 `pedestrian_attr_sophgo_out.jpg`；`vis_attr/dis_attr` 来自 `csrc/vision/common/visualize/visualize.h`、`display/display.h`。
- demo 的 `.bmodel` 尾判定：非 bmodel → `use_ort_backend()`（`ENABLE_ORT=OFF` 时该函数为空操作，仅供本地对照；纯 Sophgo 构建只走 bmodel 分支）。

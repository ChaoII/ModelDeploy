# 发布流程

本文说明 ModelDeploy 打 tag 发布到 GitHub Release 并自动上传资产(install 包 + wheel)的规范化步骤。

## 触发机制

- `build_release.yml` 在 `tags: v*` 时触发,运行 CPU/GPU 构建,并把安装包作为 **Release 资产**上传。
  - 资产文件名由 **tag 名** 驱动:`ModelDeploy_<tag>_<平台>.tar.gz/zip`。
  - CPU job 会跑 `ctest`(排除 `[opencl]`/`[vulkan]` GPU 标签);GPU job 仅编译不跑测试。
- `build_wheel.yml` 只构建 wheel 到 GitHub Artifacts(无测试),由发布者手动下载或二次上传。

## 发布步骤

1. **同步 bump 版本**(两处必须一致,参考 `.superpowers/plans/2026-08-29-v2.4.0-e.md` Task 1):
   - `VERSION_NUMBER` → `X.Y.Z`(无 `v` 前缀,如 `2.4.0`)。
   - `pyproject.toml` → `version = "X.Y.Z"`。
   - `VERSION_NUMBER` 由根 `CMakeLists.txt:52` 读为 `MD_VERSION` 编译宏;`pyproject` 决定 wheel 版本。

2. **提交到 `main`**:

   ```bash
   git add VERSION_NUMBER pyproject.toml
   git commit -m "release: bump version to X.Y.Z"
   git push origin main
   ```

3. **打 tag 并推送** 触发 `build_release.yml` 的 tag 工作流:

   ```bash
   git tag -a vX.Y.Z -m "ModelDeploy vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. **核对 Release 资产**:工作流完成后,在 GitHub Release(或 `vX.Y.Z` 页面)确认
   `ModelDeploy_vX.Y.Z_linux-x64.tar.gz`、`ModelDeploy_vX.Y.Z_win-x64.zip`(及 GPU 变体)均已上传。

> **历史欠账提醒**:`v2.3.0` 发布时曾遗留未 bump,当时 `VERSION_NUMBER`/`pyproject.toml`
> 仍为 `0.0.0`。**发布前务必核对两处版本号已同步**,避免资产名与版本号脱节。

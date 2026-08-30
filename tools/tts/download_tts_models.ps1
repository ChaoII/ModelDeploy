# Download TTS models (Audio8 + Qwen3-TTS-12Hz-0.6B) and prep dirs for later C++ porting.
# Models / big files are NOT committed to git; only this script is.
# 注意：本机 `huggingface-cli` 已废弃（只打印提示、不再下载），本脚本以新版 `hf` CLI 为主，
#       以 `modelscope` 为兜底。代理 127.0.0.1:10809 已验证可用，失败时自动直连重试。
param([string]$Root = "C:\ai\models\modeldeploy\tts")
$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

function Use-Proxy {
    param([switch]$Off)
    if ($Off) {
        Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
        $env:NO_PROXY = "127.0.0.1,localhost,::1"
        Write-Host "  [net] 直连（免代理）"
    } else {
        $env:HTTP_PROXY  = "http://127.0.0.1:10809"
        $env:HTTPS_PROXY = "http://127.0.0.1:10809"
        $env:ALL_PROXY   = "http://127.0.0.1:10809"
        Write-Host "  [net] 使用本地代理 127.0.0.1:10809"
    }
}

# 从 HF 下载（hz CLI），支持多个 --include 模式；返回 $true 表示成功。
function Invoke-HFDownload {
    param(
        [Parameter(Mandatory)][string]$RepoId,
        [Parameter(Mandatory)][string]$LocalDir,
        [string[]]$Include
    )
    if (Get-Command hf -ErrorAction SilentlyContinue) {
        $incArgs = @()
        foreach ($p in $Include) { $incArgs += "--include"; $incArgs += $p }
        & hf download $RepoId --local-dir $LocalDir @incArgs *> $null
        return ($LASTEXITCODE -eq 0)
    }
    if (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
        & huggingface-cli download $RepoId --local-dir $LocalDir @Include *> $null
        return ($LASTEXITCODE -eq 0)
    }
    return $false
}

New-Item -ItemType Directory -Force -Path $Root | Out-Null

# ---------------- Audio8 ----------------
$audio8Dir = Join-Path $Root "audio8_preview"
Write-Host "== Audio8: $audio8Dir"
if (-not (Test-Path (Join-Path $audio8Dir "runtime_manifest.json"))) {
    Use-Proxy
    if (-not (Invoke-HFDownload -RepoId "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4" -LocalDir $audio8Dir)) {
        Write-Host "  HF 失败，尝试免代理直连..."
        Use-Proxy -Off
        Invoke-HFDownload -RepoId "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4" -LocalDir $audio8Dir | Out-Null
    }
    if (-not (Test-Path (Join-Path $audio8Dir "runtime_manifest.json"))) {
        Write-Host "  HF 不可用，尝试 ModelScope 镜像..."
        if (Get-Command modelscope -ErrorAction SilentlyContinue) {
            Use-Proxy -Off
            modelscope download --model "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4" --local_dir $audio8Dir | Out-Null
        }
    }
}
if (Test-Path (Join-Path $audio8Dir "runtime_manifest.json")) {
    Write-Host "  OK audio8 manifest"
} else {
    throw "audio8 下载失败：缺 runtime_manifest.json"
}

# ---------------- Qwen3-TTS-12Hz-0.6B ----------------
$qwenDir = Join-Path $Root "qwen3_tts_0.6b"
Write-Host "== Qwen3: $qwenDir"
New-Item -ItemType Directory -Force -Path $qwenDir | Out-Null

# 社区 ONNX 全套（sherpa-onnx 命名，含 talker_prefill.onnx 等子模型）：
#   zukky/Qwen3-TTS-ONNX-DLL
#   0.6B 的 ONNX 在 onnx_kv_06b/，tokenizer 在 models/Qwen3-TTS-12Hz-0.6B-Base/
function Test-QwenOrt { param($dir) Test-Path (Join-Path $dir "onnx_kv_06b\talker_prefill.onnx") }

if (-not (Test-QwenOrt $qwenDir)) {
    Use-Proxy
    if (-not (Invoke-HFDownload -RepoId "zukky/Qwen3-TTS-ONNX-DLL" -LocalDir $qwenDir `
            -Include "onnx_kv_06b/*", "models/Qwen3-TTS-12Hz-0.6B-Base/*", "README.md")) {
        Write-Host "  社区 ONNX 未拿到，尝试免代理重试..."
        Use-Proxy -Off
        Invoke-HFDownload -RepoId "zukky/Qwen3-TTS-ONNX-DLL" -LocalDir $qwenDir `
            -Include "onnx_kv_06b/*", "models/Qwen3-TTS-12Hz-0.6B-Base/*", "README.md" | Out-Null
    }
    if (-not (Test-QwenOrt $qwenDir)) {
        Write-Host "  社区 ONNX 仍未拿到，尝试 ModelScope 同仓库名（可能不存在）..."
        if (Get-Command modelscope -ErrorAction SilentlyContinue) {
            Use-Proxy -Off
            modelscope download --model "zukky/Qwen3-TTS-ONNX-DLL" --local_dir $qwenDir | Out-Null
        }
    }
}
if (Test-QwenOrt $qwenDir) {
    Write-Host "  OK qwen3 ONNX 全套（onnx_kv_06b/ + tokenizer under models/Qwen3-TTS-12Hz-0.6B-Base/）"
} else {
    Write-Host "  [WARN] 社区 ONNX 不可用；下载官方 safetensors 作兜底（需后续转 ONNX）"
    Use-Proxy
    Invoke-HFDownload -RepoId "Qwen/Qwen3-TTS-12Hz-0.6B-Base" -LocalDir (Join-Path $Root "qwen3_checkpoint") | Out-Null
}

Write-Host "done"

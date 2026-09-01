$ErrorActionPreference = "Stop"
$root = "E:\CLionProjects\ModelDeploy"
Set-Location $root
$reg = Get-Content "tools/convert/models.json" -Raw | ConvertFrom-Json
$trtexec = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.16.1.11\bin\trtexec.exe"
$fail = 0
foreach ($m in $reg) {
    $rel = if ($m.dir) { "$($m.dir)/$($m.stem)" } else { $m.stem }
    $src = "test_data/test_models/onnx/$rel.onnx"
    $dstDir = "test_data/test_models/trt/$($m.dir)"
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    $dst = "test_data/test_models/trt/$rel.engine"
    if (Test-Path $dst) {
        if ((Get-Item $dst).Length -gt 0) { Write-Host "skip(exists): $dst"; continue }
        Remove-Item $dst -Force   # 0 字节 = trtexec 首轮失败留下的空壳，清掉触发 profile 重试
    }
    Write-Host "=== TRT: $rel ($($m.shape)) ==="
    # 先按静态(不传 profile)转：全静态模型传 min/opt/max 会导致 Network setup failed。
    & $trtexec --onnx=$src --saveEngine=$dst --fp16 2>&1 | Out-Null
    if (-not (Test-Path $dst)) {
        Write-Host "  retry with profile ($($m.input):$($m.shape))"
        & $trtexec --onnx=$src --saveEngine=$dst --fp16 `
            --minShapes="$($m.input):$($m.shape)" --optShapes="$($m.input):$($m.shape)" --maxShapes="$($m.input):$($m.shape)" 2>&1 | Out-Null
    }
    if (-not (Test-Path $dst)) { Write-Host "FAILED: $rel"; $fail++ } else { Write-Host "OK: $dst" }
}
Write-Host "TRT_DONE fail=$fail"
exit $fail

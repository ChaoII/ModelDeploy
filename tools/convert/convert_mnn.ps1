# ONNX -> MNN 批量转换（25 项，镜像 onnx 目录）。
# 依赖：MNNConvert 3.2.1（paddlex env）+ onnxsim + paddlex 的 MNN python（加载校验）。
# 坑：MNNConvert 对部分图解析不到 input（"inputTensors : [ ]"）→ 先 onnxsim 简化再转；
#     转换后还必须能 load_module 才算通过（1k3d68 曾产出损坏 .mnn）。
$ErrorActionPreference = "Stop"
$root = "E:\CLionProjects\ModelDeploy"
Set-Location $root
$reg = Get-Content "tools/convert/models.json" -Raw | ConvertFrom-Json
$cv = "C:\Users\aichao\.conda\envs\paddlex\Scripts\mnnconvert.exe"
$pyMNN = "C:\Users\aichao\.conda\envs\paddlex\python.exe"

function Test-MnnLoad([string]$mnnPath, [string]$inName) {
    $code = "import MNN, MNN.nn as nn;`n" +
            "nn.load_module_from_file(r'$mnnPath', [r'$inName'], [])`n" +
            "print('LOAD_OK')"
    $r = & $pyMNN -c $code 2>&1 | Out-String
    return ($LASTEXITCODE -eq 0) -and ($r -match 'LOAD_OK')
}

$fail = 0
foreach ($m in $reg) {
    $rel = if ($m.dir) { "$($m.dir)/$($m.stem)" } else { $m.stem }
    $src = "test_data/test_models/onnx/$rel.onnx"
    $dstDir = "test_data/test_models/mnn/$($m.dir)"
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
    $dst = "test_data/test_models/mnn/$rel.mnn"
    if (Test-Path $dst) { Write-Host "skip(exists): $dst"; continue }
    Write-Host "=== MNN: $rel ==="
    $convSrc = $src
    $tmpSim = Join-Path $env:TEMP "md_mnn_sim.onnx"
    $simPy = "import onnxsim,onnx; onnx.save(onnxsim.simplify(onnx.load(r'$($src -replace '\\','/')'))[0], r'$($tmpSim -replace '\\','/')')"
    $simR = & python -c $simPy 2>&1 | Out-String
    if (($LASTEXITCODE -eq 0) -and (Test-Path $tmpSim) -and ($simR -notmatch 'Traceback')) { $convSrc = $tmpSim }
    & $cv -f ONNX --modelFile=$convSrc --MNNModel=$dst 2>&1 | Out-String | ForEach-Object {
        $_.Split("`n") | Select-String -Pattern 'error|Error|fail|Fail' | Select-Object -First 2 }
    if (-not (Test-Path $dst)) {
        Write-Host "FAILED(no file): $rel"; $fail++; continue
    }
    if (-not (Test-MnnLoad $dst $m.input)) {
        Write-Host "FAILED(load): $rel"; Remove-Item $dst -Force; $fail++
    } else {
        Write-Host "OK: $dst"
    }
}
Write-Host "MNN_DONE fail=$fail"
exit $fail

#!/usr/bin/env pwsh
# 一键下载 ModelDeploy 测试数据(test_data.zip)并解压到仓库根 test_data/
[CmdletBinding()]
param(
    [string]$Url = "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/test_data.zip",
    [switch]$Force,
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"

# 默认输出到本脚本上级目录(仓库根)
if ([string]::IsNullOrEmpty($OutDir)) {
    $OutDir = Split-Path -Parent $PSScriptRoot
}
$testDataDir = Join-Path $OutDir "test_data"
$zipPath = Join-Path $OutDir "test_data.zip"

if ((Test-Path -LiteralPath $testDataDir) -and -not $Force) {
    Write-Host "test_data/ 已存在(用 -Force 重新下载)。"
    exit 0
}

Write-Host "下载 $Url -> $zipPath"
try {
    Invoke-WebRequest -Uri $Url -OutFile $zipPath -UseBasicParsing
} catch {
    Write-Host "下载失败: $($_.Exception.Message)"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    exit 1
}

Write-Host "解压 $zipPath -> $OutDir"
try {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $OutDir -Force
} catch {
    Write-Host "解压失败: $($_.Exception.Message)"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    exit 1
}
Remove-Item -LiteralPath $zipPath -Force
Write-Host "完成: $testDataDir"

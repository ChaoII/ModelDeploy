#!/usr/bin/env pwsh
# 校验用户文档内相对 .md 链接目标是否存在。
# 扫描范围：README.md、examples/EXAMPLES.md、docs/ 顶层与 docs/api；不含根目录 .superpowers/ 开发过程文档。
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$files = @()
if (Test-Path (Join-Path $root "docs/api")) { $files += Get-ChildItem -Path (Join-Path $root "docs/api") -Filter *.md }
$files += Get-ChildItem -Path (Join-Path $root "docs") -Filter *.md
$files += Get-Item (Join-Path $root "README.md")
$files += Get-Item (Join-Path $root "examples/EXAMPLES.md")

$broken = 0
foreach ($f in $files) {
    $text = Get-Content -LiteralPath $f.FullName -Raw
    $base = Split-Path $f.FullName -Parent
    $matches = [regex]::Matches($text, '\]\(([^)#]+?)(?:#.*)?\)')
    foreach ($m in $matches) {
        $link = $m.Groups[1].Value.Trim()
        if ($link -match '^(https?://|mailto:)') { continue }
        if ($link -match '\.(png|jpg|jpeg|gif|svg)($|\s)') { continue }  # 图片链接
        # 只把"形似路径"的目标当作链接（./x、../x、x.md、或以 / 结尾）；过滤 C++ 签名等误报
        if ($link -notmatch '^(\.{0,2}/)' -and $link -notmatch '\.md($|#)' -and $link -notmatch '/$') { continue }
        $target = [System.IO.Path]::GetFullPath((Join-Path $base $link))
        if (-not (Test-Path -LiteralPath $target)) {
            Write-Output ("BROKEN: {0} -> {1} (in {2})" -f $link, $target, $f.FullName)
            $broken++
        }
    }
}
if ($broken -gt 0) { Write-Output "共 $broken 个死链"; exit 1 }
Write-Output "链接校验通过"

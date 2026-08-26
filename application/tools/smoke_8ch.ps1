<#
.SYNOPSIS
    Surveillance 8-channel local-file integration smoke test.

    Launches `surveillance.exe --port <port> --data-dir <tmp>` with an 8-task
    config (local cam00..cam07.mp4 -> 8 .flv), starts each task via the HTTP
    REST API, measures per-channel SDK decoded/encoded frame counts over ~5s,
    and asserts each channel sustains a meaningful working rate.

    NOTE: NOT part of the automated test suite. Run manually from pwsh, e.g.:
        pwsh application/tools/smoke_8ch.ps1

    Model note: TaskConfig::validate() requires a non-empty `models` list, so a
    truly empty model list would be silently dropped by load_from_directory.
    Each task therefore carries one dummy model whose (nonexistent) path fails
    to load at pipeline start; the pipeline then falls into the "no models
    loaded (preview only)" path and encodes original frames — functionally the
    model-less preview encode exercised by the unit-test suite.
#>
[CmdletBinding()]
param(
    # root of the repo (defaults to two levels up from this script)
    [string]$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    # surveillance exe (default: build/bin/surveillance.exe under repo root)
    [string]$ExePath    = (Join-Path $RepoRoot "build\bin\surveillance.exe"),
    [int]$Port          = 18090,
    # seconds to observe after starting all tasks
    [double]$SampleSec  = 5.0,
    # per-channel decoded-frame floor: 25fps * SampleSec * factor
    [double]$FpsFloor   = 25.0,
    [double]$PassFactor = 0.6
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

function Assert-Exe {
    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "surveillance.exe not found at '$ExePath'. Build it first: cmake --build build --target surveillance"
    }
}
Assert-Exe

# ── temp data + output dirs ─────────────────────────────
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("md_smoke8_" + [Guid]::NewGuid().ToString("N"))
$dataDir = Join-Path $tmpRoot "data"
$outDir  = Join-Path $tmpRoot "out"
New-Item -ItemType Directory -Path $dataDir, $outDir -Force | Out-Null

$proc     = $null
$logPath  = Join-Path $tmpRoot "surveillance.log"
$perCh    = [ordered]@{}
$failures = [System.Collections.Generic.List[string]]::new()

function Stop-Surv {
    if ($proc -and -not $proc.HasExited) {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
        $proc.WaitForExit()
    }
}

try {
    # ── source videos (cam00..cam07 exist in test_data; add a few for the plan's
    #    duplicate-when-missing fallback, unused here since cam00..cam07 all exist)
    $videoDir = Join-Path $RepoRoot "test_data\bench_videos"
    $channels = @("cam00","cam01","cam02","cam03","cam04","cam05","cam06","cam07")
    $inputs = foreach ($c in $channels) {
        $p = Join-Path $videoDir "$c.mp4"
        if (Test-Path -LiteralPath $p) { $p } else { $inputs[-1] } # reuse last when missing
    }

    # ── build tasks.json in the data-dir format understood by
    #    PipelineManager::load_from_directory  (loads configs, does NOT auto-start)
    $dummyModel = [ordered]@{
        name                 = "dummy_placeholder";
        type                 = "detection";
        path                 = "E:/__no_such_placeholder_model__.onnx";
        backend              = "ort";
        device               = "cpu";
        confidence_threshold = 0.5;
        input_size           = @(640, 640);
        roi                  = @(0, 0, 0, 0);
        interval             = 1
    }
    $tasks = @()
    for ($i = 0; $i -lt $channels.Count; $i++) {
        $id = $channels[$i]
        $flv = Join-Path $outDir "$id.flv"
        $tasks += [ordered]@{
            id             = $id;
            name           = $id;
            input_url      = $inputs[$i];
            output_url     = $flv;
            preview_url    = "http://127.0.0.1:$Port/live/$id.flv";
            enable_preview = $true;
            topology       = "per_channel";
            decoder        = [ordered]@{
                reconnect_delay_ms = 5000;
                max_reconnects     = 2;
                timeout_us         = 10000000;
                rtsp_transport     = "tcp";
                hw_accel           = "cuda";
                device_only        = $false
            };
            encoder        = [ordered]@{
                fps          = 25;
                out_width    = 1280;
                out_height   = 720;
                out_fps      = 25;
                bitrate_kbps = 2500;
                gop          = 12;
                codec        = "libx264";
                preset       = "ultrafast";
                tune         = "zerolatency";
                format       = "flv";
                max_b_frames = 0;
                low_latency  = $true
            };
            draw           = [ordered]@{ show_label = $true; show_score = $true };
            models         = @($dummyModel)
        }
    }
    $tasksJson = [ordered]@{ tasks = $tasks } | ConvertTo-Json -Depth 10
    Set-Content -LiteralPath (Join-Path $dataDir "tasks.json") -Value $tasksJson -Encoding utf8

    # ── launch surveillance.exe ────────────────────────
    $proc = Start-Process -FilePath $ExePath `
        -ArgumentList @("--port", "$Port", "--data-dir", $dataDir) `
        -RedirectStandardOutput "$logPath.out" -RedirectStandardError "$logPath.err" `
        -PassThru -NoNewWindow

    $base = "http://127.0.0.1:$Port"
    # wait for HTTP health (up to 30s)
    $ready = $false
    for ($t = 0; $t -lt 60; $t++) {
        if ($proc.HasExited) { throw "surveillance.exe exited early. See $logPath" }
        try {
            $h = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 2
            if ($h.ok) { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw "surveillance.exe did not become ready. See $logPath" }

    # tasks are loaded from tasks.json but NOT auto-started: POST start each
    foreach ($id in $channels) {
        $r = Invoke-RestMethod -Method Post -Uri "$base/api/v1/tasks/$id/start" -TimeoutSec 5
        if (-not $r.ok) { throw "start failed for $id : $($r.msg)" }
    }

    # observe over ~SampleSec
    Start-Sleep -Seconds $SampleSec

    $threshold = [math]::Floor($FpsFloor * $SampleSec * $PassFactor)
    Write-Host "############################################################"
    Write-Host "# 8-ch smoke: exe=$ExePath  port=$Port"
    Write-Host "# window=$SampleSec s  floor@$PassFactor => per-ch floor=$threshold"
    Write-Host "############################################################"

    $allPass = $true
    foreach ($id in $channels) {
        $st = Invoke-RestMethod -Uri "$base/api/v1/tasks/$id/stats" -TimeoutSec 5
        $s  = $st.stats
        $fin  = [int64]$s.sdk_frames_in
        $fout = [int64]$s.sdk_frames_out
        $drop = [int64]$s.sdk_dropped
        $enc  = [double]$s.sdk_avg_encode_ms
        $meas = [math]::Round($fout / $SampleSec, 1)
        $ok = ($fout -ge $threshold)
        if (-not $ok) { $allPass = $false; $failures.Add($id) }
        $perCh[$id] = [ordered]@{ frames_in = $fin; frames_out = $fout; dropped = $drop; avg_encode_ms = $enc; measured_fps = $meas; pass = $ok }
        Write-Host ("  {0,-6} frames_in={1,-6} frames_out={2,-6} dropped={3,-5} enc={4,6:N2}ms  {5}" -f `
            $id, $fin, $fout, $drop, $enc, $(if ($ok) { "PASS" } else { "FAIL" }))
    }

    # dropped not exploding: report (no hard gate beyond growth)
    if ($allPass) {
        Write-Host "ALL 8 CHANNELS PASS (>= $threshold frames_out)"
        # output .flv sanity
        foreach ($id in $channels) {
            $flv = Join-Path $outDir "$id.flv"
            $sz = if (Test-Path -LiteralPath $flv) { (Get-Item -LiteralPath $flv).Length } else { 0 }
            Write-Host ("  {0,-6} flv bytes = {1}" -f $id, $sz)
        }
    } else {
        Write-Host "FAILURES: $($failures -join ', ')"
    }

    if (-not $allPass) { throw "some channels below threshold (floor=$threshold); see numbers above" }
    $script:perCh = $perCh
}
finally {
    Stop-Surv
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}

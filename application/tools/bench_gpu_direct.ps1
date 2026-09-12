# GPU-direct 多路 RTSP 基准：NVDEC 解码（设备 NV12, device_only）→ 推理（ORT TRT EP）
# → NVENC 直编（零主机拷贝）。源经 ZLMediaKit 以「URL 路径」区分（非每路一端口）。
#
# 前置：
#   - 本机 NVIDIA GPU + CUDA + TensorRT；`WITH_GPU=ON` 构建 surveillance。
#   - 运行中的 ZLMediaKit（默认 RTSP :8554）。
#   - 用 ffmpeg 把本地素材实时推流到 ZLM 的 /live/camNN（脚本自动推 N 路）。
#
# 用法：pwsh application/tools/bench_gpu_direct.ps1 -N 10
param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
  [string]$ExePath  = "",
  [string]$FFmpeg   = "ffmpeg",
  [string]$Clip     = "",
  [string]$Model    = "",
  [bool]$UseTrtEp   = $true,
  [string]$DecHw    = "cuda",
  [bool]$DeviceOnly = $true,
  [string]$DecBackend = "ffmpeg",
  [string]$EncBackend = "ffmpeg",
  [string]$EncCodec = "h264_nvenc",
  [bool]$GpuDirect  = $true,
  [bool]$Preview    = $true,
  [string]$RtspBase = "rtsp://127.0.0.1:8554/live",
  [int]$N = 10,
  [int]$Port = 18120,
  [double]$SampleSec = 10.0
)
$ErrorActionPreference = "Stop"
if (-not $ExePath) {
  foreach ($c in @("build_surv_gpu\bin\surveillance.exe","build_gpu\bin\surveillance.exe","build\bin\surveillance.exe")) {
    if (Test-Path (Join-Path $RepoRoot $c)) { $ExePath = Join-Path $RepoRoot $c; break }
  }
  if (-not $ExePath) { throw "surveillance.exe not found; build with -DWITH_GPU=ON -DBUILD_SURVEILLANCE=ON" }
}
if (-not $Clip) { $Clip = Join-Path $env:TEMP "md_bench_clip.mp4" }
if (-not (Test-Path $Clip)) {
  Write-Host "generating 20s 640x480@25 clip -> $Clip"
  & $FFmpeg -y -f lavfi -i "testsrc2=size=640x480:rate=25" -t 20 -pix_fmt yuv420p -c:v libx264 -preset ultrafast $Clip 2>$null
}
if (-not $Model) { $Model = Join-Path $RepoRoot "test_data\test_models\onnx\yolo11n\yolo11n.onnx" }

$tmp = Join-Path $env:TEMP ("md_gd_" + [Guid]::NewGuid().ToString("N").Substring(0,6))
New-Item -ItemType Directory -Force -Path (Join-Path $tmp "flv") | Out-Null
$base = "http://127.0.0.1:$Port"
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34\lib") {
  $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34\lib;" + $env:PATH
}
Get-Process surveillance -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$pubs=@(); for ($i=0; $i -lt $N; $i++) {
  $url = "$RtspBase/cam{0:d2}" -f $i
  $pubs += Start-Process -FilePath $FFmpeg -ArgumentList @("-re","-stream_loop","-1","-i",$Clip,"-c","copy","-f","rtsp","-rtsp_transport","tcp",$url) -PassThru -WindowStyle Hidden
}
Start-Sleep -Seconds 3
$proc = Start-Process -FilePath $ExePath -ArgumentList @("--port","$Port","--data-dir","$tmp") -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
function PostJson($u,$o){ Invoke-RestMethod -Uri $u -Method Post -ContentType "application/json" -Body ($o|ConvertTo-Json -Depth 8) -TimeoutSec 300 }
function GetJson($u){ Invoke-RestMethod -Uri $u -TimeoutSec 10 }
try {
  for ($i=0; $i -lt $N; $i++) {
    $id = "cam{0:d2}" -f $i
    $task = @{ id=$id; name=$id; input_url="$RtspBase/$id"; output_url=(Join-Path $tmp "flv\$id.flv"); enable_preview=$Preview;
      models=@(@{ name="yolo11n"; type="detection"; path=$Model; backend="ort"; device="gpu"; use_trt_ep=$UseTrtEp; confidence_threshold=0.3; input_size=@(640,640); roi=@(0,0,0,0); interval=1 });
      decoder=@{ backend=$DecBackend; hw_accel=$DecHw; device_only=$DeviceOnly; rtsp_transport="tcp" };
      encoder=@{ backend=$EncBackend; codec=$EncCodec; hw_accel="cuda"; gpu_direct_input=$GpuDirect; format="flv"; bitrate_kbps=2000; gop=25 } }
    PostJson "$base/api/v1/tasks" $task | Out-Null
    PostJson "$base/api/v1/tasks/$id/start" @{} | Out-Null
  }
  Start-Sleep -Seconds 4
  $a=GetJson "$base/api/v1/tasks"; $t0=Get-Date; $f0=@{}; foreach($t in $a.tasks){ $f0[$t.id]=[int64]$t.stats.frames }
  Start-Sleep -Seconds $SampleSec
  $t1=Get-Date; $b=GetJson "$base/api/v1/tasks"; $dt=($t1-$t0).TotalSeconds
  "channels=$($b.tasks.Count) decBackend=$DecBackend decHW=$DecHw deviceOnly=$DeviceOnly encBackend=$EncBackend enc=$EncCodec gpuDirect=$GpuDirect sample=$([math]::Round($dt,1))s"
  "id     fps   frames sdk_in sdk_out drop infer draw encSub total  dec(sdk) enc(sdk) err"
  $ok=0
  foreach($t in $b.tasks){
    $fr=[int64]$t.stats.frames-[int64]$f0[$t.id]; $fps=[math]::Round($fr/$dt,1); if($fps -ge 24){$ok++}; $s=$t.stats
    "{0} {1,5} {2,6} {3,6} {4,7} {5,4} {6,5} {7,4} {8,5} {9,5} {10,7} {11,6} {12}" -f $t.id,$fps,$s.frames,$s.sdk_frames_in,$s.sdk_frames_out,$s.sdk_dropped,([math]::Round($s.avg_infer_ms,1)),([math]::Round($s.avg_draw_ms,2)),([math]::Round($s.avg_encode_ms,2)),([math]::Round($s.avg_total_ms,1)),([math]::Round($s.sdk_avg_decode_ms,1)),([math]::Round($s.sdk_avg_encode_ms,1)),$t.init_error
  }
  "channels >=24fps: $ok / $($b.tasks.Count)"
} finally {
  Get-Process surveillance -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  $pubs | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

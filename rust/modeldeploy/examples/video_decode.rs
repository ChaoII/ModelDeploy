use anyhow::{bail, Result};
use modeldeploy::{Backpressure, CodecBackend, VideoCapabilities, VideoConfig, VideoDecoder, VideoFrame};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("用法: video_decode <视频文件或RTSP/RTMP地址> [帧数]");
    }
    let url = &args[1];
    let n_frames: usize = args.get(2).map(|s| s.parse().unwrap_or(10)).unwrap_or(10);

    // 探测运行环境能力
    let cap = VideoCapabilities::probe()?;
    println!(
        "能力: ffmpeg={} gstreamer={} hw_decoders={:?} hw_encoders={:?}",
        cap.ffmpeg_available()?,
        cap.gstreamer_available()?,
        cap.hw_decoders()?,
        cap.hw_encoders()?,
    );

    // 构建配置（与 C++ VideoDecoderConfig 完全对齐）
    let mut cfg = VideoConfig::new()?;
    cfg.backend(CodecBackend::FFmpeg)?
        .backpressure(Backpressure::Block)?
        .pooling(true)?;

    let dec = VideoDecoder::create(Some(&cfg))?;
    dec.open(url)?;

    let (w, h, fps) = dec.size()?;
    println!("输入: {}x{} @ {}fps", w, h, fps);

    // 同步读帧（帧为自有 Image，自动释放）
    println!("同步读取前 {} 帧...", n_frames);
    let mut count = 0usize;
    loop {
        match dec.read_frame() {
            Ok(VideoFrame { image, pts_ms }) => {
                println!(
                    "  #{} pts={}ms {}x{} dev={:?}",
                    count,
                    pts_ms,
                    image.width(),
                    image.height(),
                    image.device(),
                );
                count += 1;
                if count >= n_frames {
                    break;
                }
            }
            Err(e) => {
                println!("  EOF/停止解码: {}", e);
                break;
            }
        }
    }

    let st = dec.stats()?;
    println!(
        "统计: in={} out={} dropped={} avg_decode_ms={:.2} reconnects={}",
        st.frames_in, st.frames_out, st.dropped, st.avg_decode_ms, st.reconnect_count,
    );

    // 异步回调模式示例（跨线程投递，帧用后自动释放）
    let mut dec2 = VideoDecoder::create(Some(&cfg))?;
    dec2.set_callback(|f: VideoFrame| {
        println!(
            "  [async] pts={}ms {}x{}",
            f.pts_ms,
            f.image.width(),
            f.image.height(),
        );
    })?;
    dec2.open(url)?;
    dec2.start()?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    dec2.stop();
    println!("异步模式已停止");

    // 编码器示例（可选，若提供 -e 输出则编码）
    if let Some(idx) = args.iter().position(|a| a == "-e") {
        let out = &args[idx + 1];
        let mut ecfg = VideoConfig::new()?;
        ecfg.codec("h264")?.bitrate_kbps(2000)?.gop(30)?.fps(fps)?;
        let enc = modeldeploy::VideoEncoder::create(Some(&ecfg))?;
        enc.open(out, w, h, fps)?;
        // 用一个已读到的帧示例编码（此处无需真实循环，示意 API 形态）
        println!("编码器已就绪: {} (调用 encode(&img, pts) 写入)", out);
        drop(enc);
    }

    Ok(())
}

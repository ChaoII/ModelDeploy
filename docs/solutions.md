# 解决方案（Solutions）

解决方案层把**检测 / 跟踪 / 识别等基础能力组合成可交付的业务功能**。它们大多是纯后处理 / 统计逻辑（不直接依赖模型），消费上游检测 + 跟踪的 `TrackResult`，逐帧喂入即得到业务结果。

- 命名空间：视觉得 `modeldeploy::vision::solution`，音频为 `modeldeploy::audio::solution`
- 源码目录：`csrc/vision/solutions/`、`csrc/audio/solutions/`
- 共同输入：`std::vector<modeldeploy::vision::tracking::TrackResult>`

## 视觉解决方案

| 解决方案 | 职责 | 文档 |
|---------|------|------|
 | ObjectCounter | 目标跨线 / 区域进出计数 + 类别统计 | [object_counter.md](solutions/object_counter.md) |
 | RegionCounter | 多命名区域逐帧在区计数 | [region_counter.md](solutions/region_counter.md) |
 | QueueManager | 单区域排队长度（当前帧在区数） | [queue_manager.md](solutions/queue_manager.md) |
 | TrackZone | 区域跟踪目标过滤（ROI 子集） | [track_zone.md](solutions/track_zone.md) |
 | Heatmap | 目标位置密度热力图 | [heatmap.md](solutions/heatmap.md) |
| SpeedEstimator | 目标运动速度估计（px/s 与 m/s） | [speed_estimator.md](solutions/speed_estimator.md) |
| ParkingManager | 车位占用检测 | [parking_manager.md](solutions/parking_manager.md) |
| DistanceEstimator | 目标两两距离估计（社交距离 / 防碰撞） | [distance_estimator.md](solutions/distance_estimator.md) |
| ObjectBlur | 目标模糊（人脸 / 车牌隐私打码） | [object_blur.md](solutions/object_blur.md) |
| ObjectCropper | 按检测框裁剪目标子图 | [object_cropper.md](solutions/object_cropper.md) |
| VisionEye | 视平线透视可视化（针孔 / 鹰眼） | [vision_eye.md](solutions/vision_eye.md) |
| WorkoutMonitor | 健身动作计数（夹角阈值状态机） | [workout_monitor.md](solutions/workout_monitor.md) |

## 音频解决方案

| 解决方案 | 职责 | 文档 |
|---------|------|------|
| SpeakerDiarization | 说话人日志（VAD 分段 + embedding 聚类） | [speaker_diarization.md](solutions/speaker_diarization.md) |
| SpeakerSearch | 说话人声纹库检索（1:N） | [speaker_search.md](solutions/speaker_search.md) |
| StreamingSTT | 流式语音识别（VAD + ASR 实时转写） | [streaming_stt.md](solutions/streaming_stt.md) |
| TTSBatcher | TTS 批处理 + 长文本分块拼接 | [tts_batcher.md](solutions/tts_batcher.md) |

> 底层基础工具（`modeldeploy::vision::tool`）见 [tools.md](tools.md)。支持矩阵见 [README](../README.md)。

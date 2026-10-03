#!/usr/bin/env bash
set -euo pipefail

REGION=$1
CRF_OVERRIDE=${2:-}   # 可选：手动指定 CRF，留空则用编码器默认值

IFS=',' read X Y W H <<< "$REGION"
echo "Overlay ROI back to original video at x=$X, y=$Y"

# ============================================
# 1. 检测原视频编码器
# ============================================
ORIG_CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 input.mp4)
echo "Original video codec: $ORIG_CODEC"

case "$ORIG_CODEC" in
  h264)
    ENCODER="libx264"
    PIX_FMT="yuv420p"
    DEFAULT_CRF=18
    EXTRA_OPTS="-preset veryfast"
    ;;
  hevc|h265)
    ENCODER="libx265"
    PIX_FMT="yuv420p"
    DEFAULT_CRF=20
    EXTRA_OPTS="-preset medium"
    ;;
  av1)
    ENCODER="libsvtav1"
    PIX_FMT="yuv420p"
    DEFAULT_CRF=30
    EXTRA_OPTS="-preset 6"
    ;;
  vp9)
    ENCODER="libvpx-vp9"
    PIX_FMT="yuv420p"
    DEFAULT_CRF=30
    EXTRA_OPTS="-b:v 0 -row-mt 1 -cpu-used 2"
    ;;
  *)
    echo "⚠️ Unknown codec '$ORIG_CODEC', falling back to libx264"
    ENCODER="libx264"
    PIX_FMT="yuv420p"
    DEFAULT_CRF=18
    EXTRA_OPTS="-preset veryfast"
    ;;
esac

if [ -n "$CRF_OVERRIDE" ]; then
  CRF=$CRF_OVERRIDE
else
  CRF=$DEFAULT_CRF
fi

echo "Using encoder=$ENCODER, crf=$CRF, pix_fmt=$PIX_FMT, opts=$EXTRA_OPTS"

# ============================================
# 2. 拼接所有修复好的 ROI 分片
# ============================================
> list.txt
for part_file in parts/part-*/part_*.mp4; do
  if [ -f "$part_file" ]; then
    echo "file '$part_file'" >> list.txt
  fi
done
cat list.txt

ffmpeg -y -f concat -safe 0 -i list.txt -c copy fixed_roi.mp4

# ============================================
# 3. 贴回原视频（使用原编码器重新编码）
# ============================================
ffmpeg -y \
  -i input.mp4 \
  -i fixed_roi.mp4 \
  -filter_complex "[0:v][1:v]overlay=${X}:${Y}:format=auto" \
  -c:v $ENCODER -crf $CRF $EXTRA_OPTS \
  -pix_fmt $PIX_FMT \
  -an final_video.mp4

# ============================================
# 4. 合并原音频
# ============================================
if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 input.mp4 | grep -q audio; then
  echo "Audio detected, merging..."
  ffmpeg -y -i final_video.mp4 -i input.mp4 \
    -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
    output.mp4
else
  echo "No audio stream found, output video only."
  cp final_video.mp4 output.mp4
fi

echo "Final video: output.mp4"
echo "Final codec: $ENCODER"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt \
  -of default=noprint_wrappers=1 output.mp4

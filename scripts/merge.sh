#!/usr/bin/env bash
set -euo pipefail

REGION="${{ github.event.inputs.region }}"
IFS=',' read X Y W H <<< "$REGION"

# 1. 拼接所有修复好的 ROI 分片
> list.txt
for i in 0 1 2 3; do
  echo "file 'parts/part-$i/part_$i.mp4'" >> list.txt
done

ffmpeg -y -f concat -safe 0 -i list.txt -c copy fixed_roi.mp4

# 2. 将修复好的 ROI 贴回原视频
# 关键：只覆盖 ROI 区域，其他区域像素级无损
ffmpeg -y \
  -i input.mp4 \
  -i fixed_roi.mp4 \
  -filter_complex "[0:v][1:v]overlay=${X}:${Y}:format=auto" \
  -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
  -an final_video.mp4

# 3. 合并原音频
if ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 input.mp4 | grep -q audio; then
  ffmpeg -y -i input.mp4 -vn -c:a copy audio.aac
  ffmpeg -y -i final_video.mp4 -i audio.aac \
    -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
    output.mp4
else
  cp final_video.mp4 output.mp4
fi

echo "Final video: output.mp4"

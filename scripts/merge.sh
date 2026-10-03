#!/usr/bin/env bash
set -euo pipefail

REGION=$1
CRF=${2:-18}

IFS=',' read X Y W H <<< "$REGION"
echo "Overlay ROI back to original video at x=$X, y=$Y"

> list.txt
for part_file in parts/part-*/part_*.mp4; do
  if [ -f "$part_file" ]; then
    echo "file '$part_file'" >> list.txt
  fi
done
cat list.txt

ffmpeg -y -f concat -safe 0 -i list.txt -c copy fixed_roi.mp4

ffmpeg -y \
  -i input.mp4 \
  -i fixed_roi.mp4 \
  -filter_complex "[0:v][1:v]overlay=${X}:${Y}:format=auto" \
  -c:v libx264 -crf $CRF -preset veryfast -pix_fmt yuv420p \
  -an final_video.mp4

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

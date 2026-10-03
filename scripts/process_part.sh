#!/usr/bin/env bash
set -euo pipefail

PART=$1
INPUT_VIDEO="roi.mp4"
INPUT_MASK="dynamic_mask.mp4"
FPS=${FPS:-30}
CRF=${CRF:-18}

TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 $INPUT_VIDEO)
CORE_PER_PART=$(( (TOTAL_FRAMES + 3) / 4 ))
OVERLAP=30

CORE_START=$((PART * CORE_PER_PART))
CORE_END=$((CORE_START + CORE_PER_PART - 1))
if [ $CORE_END -ge $TOTAL_FRAMES ]; then CORE_END=$((TOTAL_FRAMES - 1)); fi

PROC_START=$((CORE_START - OVERLAP))
if [ $PROC_START -lt 0 ]; then PROC_START=0; fi

PROC_END=$((CORE_END + OVERLAP))
if [ $PROC_END -ge $TOTAL_FRAMES ]; then PROC_END=$((TOTAL_FRAMES - 1)); fi

OFFSET=$((CORE_START - PROC_START))

echo "Part $PART: Core [$CORE_START,$CORE_END], Process [$PROC_START,$PROC_END], Offset $OFFSET"

mkdir -p work/$PART
cd work/$PART

ffmpeg -y -i ../../$INPUT_VIDEO \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -c:v libx264 -crf 10 -preset veryfast -pix_fmt yuv420p seg.mp4

ffmpeg -y -i ../../$INPUT_MASK \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -c:v libx264 -crf 10 -preset veryfast -pix_fmt gray seg_mask.mp4

# 获取绝对路径，避免 ProPainter 内部路径寻址失败
SEG_VIDEO="$(pwd)/seg.mp4"
SEG_MASK="$(pwd)/seg_mask.mp4"
OUT_DIR="$(pwd)/out"
mkdir -p "$OUT_DIR"

# 切换到 ProPainter 目录下运行
cd ../../ProPainter
python inference_propainter.py \
  --video "$SEG_VIDEO" \
  --mask "$SEG_MASK" \
  --output "$OUT_DIR" \
  --subvideo_length 20 \
  --neighbor_length 5 \
  --ref_stride 5

cd -

INPAINT_VIDEO=$(find out -name "*inpaint*.mp4" | head -n 1 || true)
if [ -z "$INPAINT_VIDEO" ]; then INPAINT_VIDEO=$(find out -name "*.mp4" | head -n 1 || true); fi
if [ -z "$INPAINT_VIDEO" ]; then echo "ProPainter output not found"; exit 1; fi

ffmpeg -y -i "$INPAINT_VIDEO" \
  -vf "select='between(n,$OFFSET,$((OFFSET+CORE_PER_PART-1)))',setpts=N/FRAME_RATE/TB" \
  -vsync 0 -an -r $FPS \
  -c:v libx264 -crf $CRF -preset veryfast -pix_fmt yuv420p \
  ../../part_$PART.mp4

echo "Part $PART done: ../../part_$PART.mp4"

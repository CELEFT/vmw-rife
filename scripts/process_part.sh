#!/usr/bin/env bash
set -euo pipefail

PART=$1
INPUT_VIDEO=${INPUT_VIDEO:-roi.mp4}
INPUT_MASK=${INPUT_MASK:-dynamic_mask.mp4}
TOTAL_FRAMES=${TOTAL_FRAMES:-596}
CORE_PER_PART=${CORE_PER_PART:-149}
OVERLAP=${OVERLAP:-30}
FPS=${FPS:-30}

# 1. 计算当前分片的帧范围
CORE_START=$((PART * CORE_PER_PART))
CORE_END=$((CORE_START + CORE_PER_PART - 1))

PROC_START=$((CORE_START - OVERLAP))
if [ $PROC_START -lt 0 ]; then PROC_START=0; fi

PROC_END=$((CORE_END + OVERLAP))
if [ $PROC_END -ge $TOTAL_FRAMES ]; then PROC_END=$((TOTAL_FRAMES - 1)); fi

OFFSET=$((CORE_START - PROC_START))

echo "Part $PART: Core [$CORE_START,$CORE_END], Process [$PROC_START,$PROC_END], Offset $OFFSET"

mkdir -p work/$PART
cd work/$PART

# 2. 精准切出视频片段和Mask片段（帧范围完全一致）
ffmpeg -y -i ../../$INPUT_VIDEO \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -c:v libx264 -crf 10 -preset veryfast -pix_fmt yuv420p seg.mp4

ffmpeg -y -i ../../$INPUT_MASK \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -c:v libx264 -crf 10 -preset veryfast -pix_fmt gray seg_mask.mp4

# 3. ProPainter 修复
python ../../ProPainter/inference_propainter.py \
  --video seg.mp4 \
  --mask seg_mask.mp4 \
  --output out \
  --fp16 \
  --subvideo_length 20 \
  --neighbor_length 5 \
  --ref_stride 5

# 4. 找到输出并裁剪核心区间
INPAINT_VIDEO=$(find out -name "*inpaint*.mp4" | head -n 1 || true)
if [ -z "$INPAINT_VIDEO" ]; then
  INPAINT_VIDEO=$(find out -name "*.mp4" | head -n 1 || true)
fi

if [ -z "$INPAINT_VIDEO" ]; then
  echo "ProPainter output not found"; exit 1
fi

ffmpeg -y -i "$INPAINT_VIDEO" \
  -vf "select='between(n,$OFFSET,$((OFFSET+CORE_PER_PART-1)))',setpts=N/FRAME_RATE/TB" \
  -vsync 0 -an -r $FPS \
  -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
  ../../part_$PART.mp4

echo "Part $PART done: ../../part_$PART.mp4"

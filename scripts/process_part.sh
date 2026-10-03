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

# ============================================
# 1. 切出视频片段（视频保持 mp4 格式）
# ============================================
ffmpeg -y -i ../../$INPUT_VIDEO \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -vsync 0 -c:v libx264 -crf 10 -preset veryfast -pix_fmt yuv420p seg.mp4

# ============================================
# 2. 切出 Mask 片段（关键修复：输出为帧序列目录）
# ============================================
mkdir -p seg_mask_frames
ffmpeg -y -i ../../$INPUT_MASK \
  -vf "trim=start_frame=$PROC_START:end_frame=$((PROC_END+1)),setpts=PTS-STARTPTS" \
  -an -vsync 0 -pix_fmt gray seg_mask_frames/%06d.png

MASK_COUNT=$(ls seg_mask_frames/*.png 2>/dev/null | wc -l)
echo "Mask frames generated: $MASK_COUNT"

SEG_VIDEO="$(pwd)/seg.mp4"
SEG_MASK_DIR="$(pwd)/seg_mask_frames"
OUT_DIR="$(pwd)/out"
mkdir -p "$OUT_DIR"

# ============================================
# 3. 切换到 ProPainter 目录运行（--mask 指向帧序列目录）
# ============================================
cd ../../ProPainter
python inference_propainter.py \
  --video "$SEG_VIDEO" \
  --mask "$SEG_MASK_DIR" \
  --output "$OUT_DIR" \
  --subvideo_length 20 \
  --neighbor_length 5 \
  --ref_stride 5

cd -

INPAINT_VIDEO=$(find out -name "*inpaint*.mp4" | head -n 1 || true)
if [ -z "$INPAINT_VIDEO" ]; then INPAINT_VIDEO=$(find out -name "*.mp4" | head -n 1 || true); fi
if [ -z "$INPAINT_VIDEO" ]; then echo "ProPainter output not found"; exit 1; fi

# ============================================
# 4. 裁剪核心区间（去掉重叠的 30 帧）
# ============================================
ffmpeg -y -i "$INPAINT_VIDEO" \
  -vf "select='between(n,$OFFSET,$((OFFSET+CORE_PER_PART-1)))',setpts=N/FRAME_RATE/TB" \
  -vsync 0 -an -r $FPS \
  -c:v libx264 -crf $CRF -preset veryfast -pix_fmt yuv420p \
  ../../part_$PART.mp4

echo "Part $PART done: ../../part_$PART.mp4"

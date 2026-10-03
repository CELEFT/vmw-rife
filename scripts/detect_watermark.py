import cv2
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--total_frames', type=int, required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--sample_interval', type=int, default=15, help='每15帧采样一次')
    parser.add_argument('--min_count_ratio', type=float, default=0.6, help='频率统计阈值')
    parser.add_argument('--expand', type=int, default=10, help='膨胀像素')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. 采样检测
    sample_frames = list(range(0, args.total_frames, args.sample_interval))
    if sample_frames[-1] != args.total_frames - 1:
        sample_frames.append(args.total_frames - 1)

    accumulate = np.zeros((h, w), dtype=np.float32)
    valid_count = 0

    for fidx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Otsu 自适应阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 简单处理：若背景偏亮，取反（假设水印通常与背景有对比度）
        if gray.mean() > 128:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        accumulate += (binary > 0).astype(np.float32)
        valid_count += 1

    # 2. 频率统计生成稳定Mask
    min_count = valid_count * args.min_count_ratio
    stable_mask = np.where(accumulate >= min_count, 255, 0).astype(np.uint8)

    # 膨胀覆盖边缘
    kernel = np.ones((5, 5), np.uint8)
    stable_mask = cv2.dilate(stable_mask, kernel, iterations=1)

    # 3. 生成逐帧Mask视频（在整个视频范围内固定不变）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h), isColor=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(args.total_frames):
        out.write(stable_mask)
        
    cap.release()
    out.release()
    print(f"Dynamic mask generated: {args.output}")

if __name__ == '__main__':
    main()

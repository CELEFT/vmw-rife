import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--sample_interval', type=int, default=15, help='每15帧采样一次')
    parser.add_argument('--min_count_ratio', type=float, default=0.6, help='频率统计阈值')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"ROI Video: {w}x{h}, {total_frames} frames")
    
    sample_frames = list(range(0, total_frames, args.sample_interval))
    if sample_frames[-1] != total_frames - 1:
        sample_frames.append(total_frames - 1)

    accumulate = np.zeros((h, w), dtype=np.float32)
    valid_count = 0

    for fidx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if gray.mean() > 128:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        accumulate += (binary > 0).astype(np.float32)
        valid_count += 1

    min_count = valid_count * args.min_count_ratio
    stable_mask = np.where(accumulate >= min_count, 255, 0).astype(np.uint8)

    # 检查自动检测结果是否有效
    white_pixels = np.sum(stable_mask == 255)
    total_pixels = w * h
    white_ratio = white_pixels / total_pixels

    print(f"Auto-detected white pixel ratio: {white_ratio:.2%}")

    # 兜底逻辑：如果自动检测几乎全黑（小于1%）或几乎全白（大于90%），说明检测失败
    if white_ratio < 0.01 or white_ratio > 0.9:
        print("⚠️ Auto-detection failed or abnormal. Applying full ROI rectangle as fallback mask.")
        # 将整个 ROI 区域设为白色
        stable_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        # 只有在检测有效时，才进行膨胀
        kernel = np.ones((5, 5), np.uint8)
        stable_mask = cv2.dilate(stable_mask, kernel, iterations=1)

    # 生成逐帧Mask视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h), isColor=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total_frames):
        out.write(stable_mask)
        
    cap.release()
    out.release()
    print(f"Dynamic mask generated: {args.output}")

if __name__ == '__main__':
    main()

import sys
import os
import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

# ---- 参数 ----
x, y, w, h = map(int, os.environ['REGION'].split(','))
W = int(os.environ['W'])
H = int(os.environ['H'])
FRAME_SIZE = W * H * 3

# ---- 加边距给 LaMa 更多上下文 ----
MARGIN = 64
x1 = max(0, x - MARGIN)
y1 = max(0, y - MARGIN)
x2 = min(W, x + w + MARGIN)
y2 = min(H, y + h + MARGIN)

mx1 = x - x1
my1 = y - y1
mx2 = mx1 + w
my2 = my1 + h

print(f"[info] Video {W}x{H}, region=({x},{y},{w},{h})", file=sys.stderr)
print(f"[info] Crop ({x1},{y1})->({x2},{y2}), mask=({mx1},{my1})->({mx2},{my2})", file=sys.stderr)

# ---- 初始化 LaMa ----
print("[info] Loading LaMa model...", file=sys.stderr)
simple_lama = SimpleLama()
print("[info] LaMa model loaded.", file=sys.stderr)

# ---- 预构造 mask（每帧复用）----
mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
mask[my1:my2, mx1:mx2] = 255

# ---- 横向分块参数 ----
# LaMa 内部会把输入 pad 成正方形再 resize，宽扁区域会导致高度被严重压缩。
# 把宽扁 crop 切成多个近似方形的子块分别修复，质量显著提升。
N_BLOCKS = 4
OVERLAP = 96
BLOCK_W = (x2 - x1 + N_BLOCKS - 1) // N_BLOCKS


def inpaint_crop(crop_bgr, full_mask):
    """
    对 crop 分块修复：
    1. 每块扩展到 [core_x1 - OVERLAP, core_x2 + OVERLAP]
    2. 在扩展块上跑 LaMa
    3. 只把核心块中 mask 覆盖的区域贴回，避免重叠区覆盖
    """
    h_crop, w_crop = crop_bgr.shape[:2]
    result = crop_bgr.copy()

    for i in range(N_BLOCKS):
        core_x1 = i * BLOCK_W
        core_x2 = min(w_crop, core_x1 + BLOCK_W)
        if core_x1 >= w_crop:
            break

        core_mask = full_mask[:, core_x1:core_x2]
        if core_mask.max() == 0:
            continue

        ext_x1 = max(0, core_x1 - OVERLAP)
        ext_x2 = min(w_crop, core_x2 + OVERLAP)
        ext_w = ext_x2 - ext_x1

        ext_crop = crop_bgr[:, ext_x1:ext_x2].copy()
        ext_mask = np.zeros((h_crop, ext_w), dtype=np.uint8)
        ext_mask[:, core_x1 - ext_x1: core_x2 - ext_x1] = core_mask

        ext_crop_pil = Image.fromarray(cv2.cvtColor(ext_crop, cv2.COLOR_BGR2RGB))
        ext_mask_pil = Image.fromarray(ext_mask)

        fixed_pil = simple_lama(ext_crop_pil, ext_mask_pil)
        fixed = cv2.cvtColor(np.array(fixed_pil), cv2.COLOR_RGB2BGR)

        # 只把核心块中 mask 覆盖的像素贴回
        core_fixed = fixed[:, core_x1 - ext_x1: core_x2 - ext_x1]
        core_mask_bool = core_mask > 0

        # 局部变量写回，避免链式索引可读性问题
        block_dst = result[:, core_x1:core_x2]
        block_dst[core_mask_bool] = core_fixed[core_mask_bool]

    return result


# ---- 精确读取 ----
def read_exact(stream, n):
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        chunk = stream.read(n - pos)
        if not chunk:
            return None
        view[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
    return bytes(buf)


stdin = sys.stdin.buffer
stdout = sys.stdout.buffer

idx = 0
while True:
    raw = read_exact(stdin, FRAME_SIZE)
    if raw is None:
        break

    # ⚠️ .copy() 让数组可写
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3)).copy()

    crop = frame[y1:y2, x1:x2].copy()
    result = inpaint_crop(crop, mask)
    frame[y1:y2, x1:x2] = result

    stdout.write(frame.tobytes())
    stdout.flush()

    idx += 1
    if idx % 10 == 0:
        print(f"[progress] {idx} frames", file=sys.stderr)

print(f"[done] Total {idx} frames processed", file=sys.stderr)

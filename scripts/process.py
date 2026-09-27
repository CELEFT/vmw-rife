import sys
import os
import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

x, y, w, h = map(int, os.environ['REGION'].split(','))
W = int(os.environ['W'])
H = int(os.environ['H'])
FRAME_SIZE = W * H * 3

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
print(f"[info] Crop ({x1},{y1})->({x2},{y2})", file=sys.stderr)

simple_lama = SimpleLama()
print("[info] LaMa model loaded.", file=sys.stderr)

mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
mask[my1:my2, mx1:mx2] = 255

N_BLOCKS = 4
OVERLAP = 96
BLOCK_W = (x2 - x1 + N_BLOCKS - 1) // N_BLOCKS


def lama_fix(crop_bgr, mask_u8):
    h_in, w_in = crop_bgr.shape[:2]
    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_u8)
    fixed_pil = simple_lama(crop_pil, mask_pil)
    fixed = cv2.cvtColor(np.array(fixed_pil), cv2.COLOR_RGB2BGR)
    if fixed.shape[0] != h_in or fixed.shape[1] != w_in:
        print(f"[warn] LaMa {fixed.shape[:2]} != {crop_bgr.shape[:2]}, resizing", file=sys.stderr)
        fixed = cv2.resize(fixed, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
    return fixed


def inpaint_crop(crop_bgr, full_mask):
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
        fixed = lama_fix(ext_crop, ext_mask)
        core_fixed = fixed[:, core_x1 - ext_x1: core_x2 - ext_x1]
        core_mask_bool = core_mask > 0
        block_dst = result[:, core_x1:core_x2]
        block_dst[core_mask_bool] = core_fixed[core_mask_bool]
    return result


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
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3)).copy()
    crop = frame[y1:y2, x1:x2].copy()
    result = inpaint_crop(crop, mask)
    frame[y1:y2, x1:x2] = result
    stdout.write(frame.tobytes())
    stdout.flush()
    idx += 1
    if idx % 10 == 0:
        print(f"[progress] {idx}", file=sys.stderr)

print(f"[done] {idx} frames", file=sys.stderr)

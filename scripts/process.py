import sys
import os
import cv2
import numpy as np
from PIL import Image
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, HDStrategy

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
model = ModelManager(name="lama", device="cpu")
config = InpaintRequest(
    hd_strategy=HDStrategy.ORIGINAL,
    hd_strategy_crop_margin=64,
    hd_strategy_crop_trigger_size=1024,
    hd_strategy_resize_limit=2048,
)

# ---- 预构造 mask（每帧复用）----
mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
mask[my1:my2, mx1:mx2] = 255
mask_pil = Image.fromarray(mask)

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

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
    crop = frame[y1:y2, x1:x2].copy()
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    result_pil = model(crop_pil, mask_pil, config)
    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    frame[y1:y2, x1:x2] = result
    stdout.write(frame.tobytes())
    stdout.flush()

    idx += 1
    if idx % 10 == 0:
        print(f"[progress] {idx} frames", file=sys.stderr)

print(f"[done] Total {idx} frames processed", file=sys.stderr)

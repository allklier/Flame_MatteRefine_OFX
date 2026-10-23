# Crop strip: input / ViTMatte-small / DiffMatte ViT-B s1 / DiffMatte ViT-S-1024 s1 / truth.
import sys, glob, numpy as np, cv2
Q, scene, out = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 1920, 1080; x0, y0, cw, ch = 700, 300, 520, 340
def ld(model, k):
    return np.fromfile(glob.glob(f"{Q}/{model}/{scene}/*_{k}.f32")[0], np.float32).reshape(H, W)
tiles = [ld("vitmatte-small", "in"), ld("vitmatte-small", "ml"), ld("dm-vitb-s1", "ml"),
         ld("dm-vits1024-s1", "ml"), ld("vitmatte-small", "truth")]
row = np.hstack([np.pad(t[y0:y0+ch, x0:x0+cw], 2, constant_values=0.5) for t in tiles])
cv2.imwrite(out, (np.clip(row, 0, 1) * 255).astype(np.uint8))

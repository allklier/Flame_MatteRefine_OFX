# Wrapper (Tile) vs the authors' ddim_sample on the repo's demo image (crop 512) and on random
# input; prints max |diff| and a few alpha stats as a sanity check of the loaded weights.
import sys, numpy as np, torch, cv2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import dm_common as dc
variant, wts, img_path, tri_path = sys.argv[1:5]
m = dc.build(variant, wts)
T = 512
im = cv2.imread(img_path)[:, :, ::-1].astype(np.float32) / 255
tr = cv2.imread(tri_path, 0).astype(np.float32) / 255
print("demo image", im.shape)
tr = np.where(tr > 0.9, 1.0, np.where(tr >= 0.1, 0.5, 0.0)).astype(np.float32)
h, w = tr.shape
ys, xs = np.nonzero(tr == 0.5); cy, cx = int(ys.mean()), int(xs.mean())
y0, x0 = max(0, min(h - T, cy - T // 2)), max(0, min(w - T, cx - T // 2))
im, tr = im[y0:y0 + T, x0:x0 + T], tr[y0:y0 + T, x0:x0 + T]
rgb = torch.from_numpy(im.copy()).permute(2, 0, 1)[None]
tri = torch.from_numpy(tr.copy())[None, None]
for steps in (1, 10):
    noise = dc.make_noise(T, 0)
    ref = dc.reference(m, steps, rgb, tri, noise)
    tile = dc.Tile(m, steps, noise).eval()
    with torch.no_grad():
        out = tile(torch.cat([(rgb - 0.5) / 0.5, tri], 1))
    unk = tri == 0.5
    print(f"steps {steps}: max|wrapper-ref| {float((out-ref).abs().max()):.2e}; unknown px {int(unk.sum())}, "
          f"alpha in unknown mean {float(out[unk].mean()):.3f}, frac>0.9 {float((out[unk]>0.9).float().mean()):.3f}, "
          f"frac<0.1 {float((out[unk]<0.1).float().mean()):.3f}")
    cv2.imwrite(f"{sys.argv[5]}/demo_{variant}_s{steps}.png", (out[0, 0].numpy() * 255).astype(np.uint8))
cv2.imwrite(f"{sys.argv[5]}/demo_rgb.png", (im[:, :, ::-1] * 255).astype(np.uint8))
cv2.imwrite(f"{sys.argv[5]}/demo_tri.png", (tr * 255).astype(np.uint8))

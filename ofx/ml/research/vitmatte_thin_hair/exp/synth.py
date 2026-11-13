"""Synthetic thin-strand matting test for ViTMatte (numpy + scipy only).

  python synth.py gen  <scene_id> <jobdir>   -> job_*.in (vm_run input) + meta.npz
  python synth.py eval <jobdir> <results.csv> -> appends one row per variant
  python synth.py list                        -> number of scenes

A scene (C x C px, linear light): a solid dark mass on the left (alpha 1), two hair locks, each a pair of
anti-aliased strands of width w with an edge-to-edge gap g; lock A starts inside the mass, lock B
starts 15 px away from it. The picture is composited in linear light, noise added, then encoded the
way a variant asks (default: clamp + gamma 1/2.4, as Flame's Rec.1886 conversion, decision 34).
Trimap: fg = mass eroded by 2 px; unknown = a 'focus region' painted along each lock (circles of
radius R along its centre line, or a box); bg = everything else.
"""
import csv
import os
import sys

import numpy as np
from scipy import ndimage

C = 192  # canvas, a multiple of 32
WIDTHS = [1.5, 2.0, 3.0, 4.0, 6.0]
GAPS = [1.0, 2.0, 3.0]
LOOKS = ["dark_skin", "dark_texture", "dark_shadow", "light_skin"]
SS = 4  # supersampling for coverage


def scenes():
    return [(look, w, g) for look in LOOKS for w in WIDTHS for g in GAPS]


def bezier(p0, p1, p2, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def offset_curve(pts, d):
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    return pts + d * normal


def dist_to_polyline(xs, ys, pts):
    """Exact distance from each (x, y) to a dense polyline (segments)."""
    q = np.stack([xs.ravel(), ys.ravel()], 1)
    best = np.full(len(q), np.inf)
    a, b = pts[:-1], pts[1:]
    for i in range(0, len(a), 64):
        aa, bb = a[i:i + 64], b[i:i + 64]
        ab = bb - aa
        t = ((q[:, None, :] - aa[None]) * ab[None]).sum(-1) / ((ab ** 2).sum(-1)[None] + 1e-12)
        t = np.clip(t, 0, 1)
        proj = aa[None] + t[..., None] * ab[None]
        best = np.minimum(best, np.sqrt(((q[:, None, :] - proj) ** 2).sum(-1)).min(1))
    return best.reshape(xs.shape)


def coverage(curves, width):
    """Box-filtered coverage of strands of `width` around each curve, supersampled SS x SS."""
    o = (np.arange(SS) + 0.5) / SS
    ys, xs = np.mgrid[0:C * SS, 0:C * SS].astype(np.float64)
    xs = (xs + 0.5) / SS
    ys = (ys + 0.5) / SS
    hit = np.zeros((C * SS, C * SS), bool)
    for c in curves:
        # only evaluate near the curve's bounding box
        x0, y0 = np.floor(c.min(0) - width - 2).astype(int)
        x1, y1 = np.ceil(c.max(0) + width + 2).astype(int)
        x0, y0 = max(x0, 0) * SS, max(y0, 0) * SS
        x1, y1 = min(x1, C) * SS, min(y1, C) * SS
        d = dist_to_polyline(xs[y0:y1, x0:x1], ys[y0:y1, x0:x1], c)
        hit[y0:y1, x0:x1] |= d <= width / 2
    del o
    return hit.reshape(C, SS, C, SS).mean((1, 3))


def make_scene(look, w, g, seed):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:C, 0:C].astype(np.float64)
    # solid mass: left side with a wavy boundary at x ~ 50
    edge = 50 + 6 * np.sin(ys / 23.0 + rng.uniform(0, 6))
    mass = np.clip(edge - xs + 0.5, 0, 1)
    # two locks; each = 2 strands, centre lines offset by +-(w + g) / 2
    sep = (w + g) / 2
    locks = []
    ya, yb = 60 + rng.uniform(-5, 5), 135 + rng.uniform(-5, 5)
    ea = 50 + 6 * np.sin(ya / 23.0)
    for (ystart, xstart) in [(ya, ea - 4), (yb, 50 + 6 * np.sin(yb / 23.0) + 15)]:
        p0 = np.array([xstart, ystart])
        p2 = p0 + np.array([95 + rng.uniform(-5, 5), rng.uniform(-25, 25)])
        p1 = (p0 + p2) / 2 + np.array([0, rng.uniform(-18, 18)])
        centre = bezier(p0, p1, p2)
        locks.append(centre)
    strands = [offset_curve(c, s) for c in locks for s in (-sep, sep)]
    hair = coverage(strands, w)
    alpha_strands = hair
    alpha = np.maximum(mass, alpha_strands)

    # colours (linear light, roughly ACEScg-like magnitudes)
    skin = np.array([0.36, 0.22, 0.16])
    if look == "light_skin":
        hair_rgb = np.array([0.30, 0.20, 0.10])
    else:
        hair_rgb = np.array([0.035, 0.022, 0.015])
    bg = skin[None, None, :] * (1 + 0.12 * (xs / C - 0.5))[..., None]
    low = ndimage.gaussian_filter(rng.normal(0, 1, (C, C)), 12)
    bg = bg * (1 + 0.08 * low / (low.std() + 1e-9))[..., None]
    fine_amp = 0.12 if look == "dark_texture" else 0.03
    fine = ndimage.gaussian_filter(rng.normal(0, 1, (C, C)), 0.8)
    bg = bg * (1 + fine_amp * fine / (fine.std() + 1e-9))[..., None]
    if look == "dark_shadow":
        bg = bg * 0.15
    hn = ndimage.gaussian_filter(rng.normal(0, 1, (C, C)), 2)
    fg = hair_rgb[None, None, :] * (1 + 0.15 * hn / (hn.std() + 1e-9))[..., None]
    fg = np.clip(fg, 0, None)
    img = alpha[..., None] * fg + (1 - alpha[..., None]) * bg
    img = img + rng.normal(0, 0.004, img.shape) * np.sqrt(np.clip(img, 1e-4, None) / 0.2)
    img = np.clip(img, 0, None)

    # trimap pieces
    fg_core = ndimage.binary_erosion(mass > 0.5, iterations=2)
    return dict(img=img.astype(np.float32), alpha=alpha.astype(np.float32), mass=mass.astype(np.float32),
                strands=alpha_strands.astype(np.float32), fg_core=fg_core, locks=locks, sep=sep, w=w, g=g)


def focus_region(locks, kind, R):
    """Unknown zone painted by an artist: circles of radius R along each lock's centre line, or a box."""
    ys, xs = np.mgrid[0:C, 0:C].astype(np.float64)
    u = np.zeros((C, C), bool)
    soft = np.zeros((C, C))
    for c in locks:
        if kind == "box":
            x0, y0 = c.min(0) - R
            x1, y1 = c.max(0) + R
            u |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        else:
            d = dist_to_polyline(xs, ys, c[::8])
            u |= d <= R
            soft = np.maximum(soft, np.clip((R + 4 - d) / 8, 0, 1))  # a soft roto edge 8 px wide
    return u, soft


def encode(img, kind):
    x = np.clip(img, 0, None)
    if kind == "g24":
        return np.clip(x, 0, 1) ** (1 / 2.4)
    if kind == "srgb":
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    if kind == "lin":
        return np.clip(x, 0, 1)
    if kind == "cct":  # ACEScct curve
        y = np.where(x <= 0.0078125, 10.5402377416545 * x + 0.0729055341958355, (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52)
        return np.clip(y, 0, 1)
    raise ValueError(kind)


IMNET_MEAN = np.array([0.485, 0.456, 0.406])
IMNET_STD = np.array([0.229, 0.224, 0.225])


def pack(rgb, tri, norm):
    if norm == "half":
        x = (rgb - 0.5) / 0.5
    else:
        x = (rgb - IMNET_MEAN) / IMNET_STD
    return np.concatenate([x.transpose(2, 0, 1), tri[None]], 0).astype(np.float32)


def up(a, s, order):
    if s == 1:
        return a
    if a.ndim == 3:
        return np.stack([ndimage.zoom(a[..., c], s, order=order, mode="nearest", grid_mode=True) for c in range(a.shape[2])], -1)
    return ndimage.zoom(a, s, order=order, mode="nearest", grid_mode=True)


def unsharp(rgb, amount=1.0, sigma=1.0):
    b = np.stack([ndimage.gaussian_filter(rgb[..., c], sigma) for c in range(3)], -1)
    return np.clip(rgb + amount * (rgb - b), 0, 1)


def local_stretch(rgb, region, out_lo=0.05, out_hi=0.95):
    """Map the 1st-99th percentile of the region's luminance to out_lo..out_hi (one gain for all channels)."""
    lum = rgb.mean(-1)
    lo, hi = np.percentile(lum[region], [1, 99])
    return np.clip((rgb - lo) / max(hi - lo, 1e-3) * (out_hi - out_lo) + out_lo, 0, 1)


D4 = [(k, f) for k in range(4) for f in (False, True)]


def d4(a, k, f):
    a = np.rot90(a, k, axes=(0, 1))
    return a[:, ::-1] if f else a


def d4_inv(a, k, f):
    if f:
        a = a[:, ::-1]
    return np.rot90(a, -k, axes=(0, 1))


# variant name -> settings. scale s, encoding, normalisation, trimap (kind, R, soft), tta, filter
VARIANTS = {
    "base": dict(),
    "imnet": dict(norm="imnet"),
    "srgb": dict(enc="srgb"),
    "lin": dict(enc="lin"),
    "cct": dict(enc="cct"),
    "up2": dict(s=2),
    "up3": dict(s=3),
    "up4": dict(s=4),
    "up2_imnet": dict(s=2, norm="imnet"),
    "tta8": dict(tta=True),
    "up2_tta8": dict(s=2, tta=True),
    "unsharp": dict(filt="unsharp"),
    "up2_unsharp": dict(s=2, filt="unsharp"),
    "stretch": dict(filt="stretch"),
    "up2_stretch": dict(s=2, filt="stretch"),
    "R6": dict(R=6),
    "R20": dict(R=20),
    "box": dict(kind="box", R=10),
    "soft": dict(soft=True),
    "up2_R20": dict(s=2, R=20),
    "ctx512": dict(ctx=512),
}
# round 2: combinations on top of the ImageNet normalisation (SET=2)
VARIANTS2 = {
    "base": dict(),
    "imnet": dict(norm="imnet"),
    "imnet_unsharp": dict(norm="imnet", filt="unsharp"),
    "imnet_unsharp2": dict(norm="imnet", filt="unsharp2"),
    "imnet_stretch": dict(norm="imnet", filt="stretch"),
    "imnet_mild": dict(norm="imnet", filt="mild"),
    "imnet_stretch_unsharp": dict(norm="imnet", filt="stretch_unsharp"),
    "imnet_lin": dict(norm="imnet", enc="lin"),
    "imnet_R6": dict(norm="imnet", R=6),
    "imnet_R20": dict(norm="imnet", R=20),
    "imnet_tta8": dict(norm="imnet", tta=True),
    "imnet_soft": dict(norm="imnet", soft=True),
    "imnet_up2": dict(norm="imnet", s=2),
}
if os.environ.get("SET") == "2":
    VARIANTS = VARIANTS2


def settings(v):
    d = dict(s=1, enc="g24", norm="half", kind="circles", R=12, soft=False, tta=False, filt=None, ctx=0)
    d.update(VARIANTS[v])
    return d


def gen(scene_id, jobdir):
    look, w, g = scenes()[scene_id]
    sc = make_scene(look, w, g, seed=1000 + scene_id)
    os.makedirs(jobdir, exist_ok=True)
    regions = {}
    for v in VARIANTS:
        st = settings(v)
        key = (st["kind"], st["R"])
        if key not in regions:
            regions[key] = focus_region(sc["locks"], st["kind"], st["R"])
        u, soft = regions[key]
        u = u & ~sc["fg_core"]
        tri = np.zeros((C, C), np.float32)
        tri[u] = 0.5
        if st["soft"]:  # soft roto value 0..0.5 at the region's edge instead of a hard step
            tri = np.where(sc["fg_core"], 0, 0.5 * soft).astype(np.float32)
        tri[sc["fg_core"]] = 1.0
        rgb = encode(sc["img"], st["enc"])
        if st["filt"] == "unsharp":
            rgb = unsharp(rgb)
        elif st["filt"] == "unsharp2":
            rgb = unsharp(rgb, amount=2.0)
        elif st["filt"] == "stretch":
            rgb = local_stretch(rgb, u)
        elif st["filt"] == "mild":
            rgb = 0.5 * rgb + 0.5 * local_stretch(rgb, u)
        elif st["filt"] == "stretch_unsharp":
            rgb = unsharp(local_stretch(rgb, u))
        s = st["s"]
        rgb_s = np.clip(up(rgb, s, 3), 0, 1)
        tri_s = up(tri, s, 0)
        if st["soft"]:
            tri_s = up(tri, s, 1)
        tfs = D4 if st["tta"] else [(0, False)]
        for i, (k, f) in enumerate(tfs):
            x = pack(d4(rgb_s, k, f), d4(tri_s, k, f), st["norm"])
            if st["ctx"]:
                P = (st["ctx"] - x.shape[1]) // 2
                img_p = np.pad(x[:3], ((0, 0), (P, P), (P, P)), mode="reflect")
                x = np.concatenate([img_p, np.pad(x[3:], ((0, 0), (P, P), (P, P)))], 0)
            H, W = x.shape[1:]
            with open(os.path.join(jobdir, f"job_{v}__{i}.in"), "wb") as fh:
                np.array([H, W], np.int32).tofile(fh)
                x.tofile(fh)
    np.savez_compressed(os.path.join(jobdir, "meta.npz"), alpha=sc["alpha"], strands=sc["strands"],
                        mass=sc["mass"], fg_core=sc["fg_core"], scene=np.array([scene_id, w, g]),
                        look=np.array(look), sep=np.array(sc["sep"]),
                        **{f"u_{k}_{R}": regions[(k, R)][0] for (k, R) in regions},
                        **{f"lock{i}": c for i, c in enumerate(sc["locks"])})


def read_out(path, shape):
    return np.fromfile(path, np.float32).reshape(shape)


def zone_masks(m):
    """Gap pixels (between the two strands of a lock, gt ~ 0) and pixels > 2 px from any strand."""
    ys, xs = np.mgrid[0:C, 0:C].astype(np.float64)
    sep, w = float(m["sep"]), float(m["scene"][1])
    gap = np.zeros((C, C), bool)
    for key in ("lock0", "lock1"):
        d = dist_to_polyline(xs, ys, m[key][::4])
        gap |= d < max(sep - w / 2 - 0.25, 0)
    near = ndimage.binary_dilation(m["strands"] > 0.02, iterations=2)
    return gap, near


def metrics(pred, m, u, gap_all, near):
    gt = m["alpha"]
    strands = m["strands"]
    sel = u & (m["mass"] < 0.01)  # unknown pixels away from the mass: the hair zone
    core = sel & (strands > 0.9)
    gap = gap_all & sel & (gt < 0.05)
    leak = sel & ~near
    pa, ga = pred[sel] > 0.5, gt[sel] > 0.5
    iou = (pa & ga).sum() / max((pa | ga).sum(), 1)
    return dict(
        cover=float(pred[core].mean()) if core.any() else np.nan,  # mean alpha on true strand cores
        iou=float(iou),
        sad=float(np.abs(pred - gt)[sel].sum() / 100.0),  # SAD / 100 in the hair zone
        gapA=float(pred[gap].mean()) if gap.any() else np.nan,  # false alpha between strands
        sep_ok=float(((pred[gap] < 0.5).mean())) if gap.any() else np.nan,
        leak=float(pred[leak].mean()) if leak.any() else np.nan,  # alpha far (> 2 px) from any strand
        n_core=int(core.sum()), n_gap=int(gap.sum()))


def evaluate(jobdir, out_csv):
    m = dict(np.load(os.path.join(jobdir, "meta.npz")))
    gap_all, near = zone_masks(m)
    rows = []
    for v in VARIANTS:
        st = settings(v)
        s = st["s"]
        tfs = D4 if st["tta"] else [(0, False)]
        acc = np.zeros((C, C))
        for i, (k, f) in enumerate(tfs):
            p = os.path.join(jobdir, f"job_{v}__{i}.out")
            if st["ctx"]:
                P = (st["ctx"] - C) // 2
                a = read_out(p, (st["ctx"], st["ctx"]))[P:P + C, P:P + C]
            else:
                a = read_out(p, (C * s, C * s))
            a = d4_inv(a, k, f)
            acc += a.reshape(C, s, C, s).mean((1, 3))  # area downsample
        pred = np.clip(acc / len(tfs), 0, 1)
        u = m[f"u_{st['kind']}_{st['R']}"] & ~m["fg_core"]
        # evaluate every variant on the same zone (the default R=12 region) for comparability, plus own zone
        u_ref = m["u_circles_12"] & ~m["fg_core"]
        r = metrics(pred, m, u_ref, gap_all, near)
        r.update(variant=v, look=str(m["look"]), w=float(m["scene"][1]), g=float(m["scene"][2]),
                 leak_own=metrics(pred, m, u, gap_all, near)["leak"])
        rows.append(r)
        np.save(os.path.join(jobdir, f"pred_{v}.npy"), pred.astype(np.float16))
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        print(len(scenes()))
    elif cmd == "gen":
        gen(int(sys.argv[2]), sys.argv[3])
    elif cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3])
    elif cmd == "oracle":  # write ground truth (or zeros) as outputs: checks the evaluator
        jobdir, mode = sys.argv[2], sys.argv[3]
        m = dict(np.load(os.path.join(jobdir, "meta.npz")))
        for fn in os.listdir(jobdir):
            if fn.endswith(".in"):
                v, i = fn[4:-3].split("__")
                st = settings(v)
                s = st["s"]
                k, f = (D4 if st["tta"] else [(0, False)])[int(i)]
                a = np.kron(m["alpha"], np.ones((s, s))) if mode == "gt" else np.zeros((C * s, C * s))
                d4(a, k, f).astype(np.float32).tofile(os.path.join(jobdir, fn[:-3] + ".out"))

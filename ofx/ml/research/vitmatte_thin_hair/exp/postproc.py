"""Post-processing of saved predictions: python postproc.py <outdir> <variant> [<variant2> ...]
Each rule is applied inside the R=12 region, then scored with synth.metrics (same zone as the main table)."""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage

import synth

out, variants = sys.argv[1], sys.argv[2:]


def levels(p, lo, hi):
    return np.clip((p - lo) / (hi - lo), 0, 1)


def area_filter(p, t=0.2, min_px=12):
    lab, n = ndimage.label(p > t)
    if n == 0:
        return p
    sizes = ndimage.sum(np.ones_like(p), lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return np.where(keep[lab], p, 0.0)


def elongation_filter(p, t=0.2, min_len=10):
    """Keep components whose bounding-box diagonal is >= min_len (hair is long and thin)."""
    lab, n = ndimage.label(p > t, structure=np.ones((3, 3)))
    if n == 0:
        return p
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        keep[i] = np.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= min_len
    return np.where(keep[lab], p, 0.0)


RULES = {
    "raw": lambda p: p,
    "cut0.05": lambda p: np.where(p < 0.05, 0, p),
    "cut0.10": lambda p: np.where(p < 0.10, 0, p),
    "levels0.05-0.8": lambda p: levels(p, 0.05, 0.8),
    "levels0.1-0.7": lambda p: levels(p, 0.1, 0.7),
    "area12": lambda p: area_filter(p),
    "elong10": lambda p: elongation_filter(p),
    "elong10+levels0.05-0.8": lambda p: levels(elongation_filter(p), 0.05, 0.8),
}

acc = defaultdict(lambda: defaultdict(list))
for d in sorted(os.listdir(out)):
    if not d.startswith("scene_"):
        continue
    m = dict(np.load(os.path.join(out, d, "meta.npz")))
    gap, near = synth.zone_masks(m)
    u = m["u_circles_12"] & ~m["fg_core"]
    look = str(m["look"])
    for v in variants:
        p = np.load(os.path.join(out, d, f"pred_{v}.npy")).astype(np.float64)
        for name, f in RULES.items():
            q = np.where(u, f(np.where(u, p, 0)), p)
            r = synth.metrics(q, m, u, gap, near)
            for k in ("cover", "iou", "sad", "gapA", "leak"):
                acc[(v, name)][k].append(r[k])
                acc[(v, name)][k + "_" + look].append(r[k])

print("| variant | rule | cover | IoU | SAD/100 | gap alpha | leak | SAD shadow | SAD light |")
print("|---|---|---|---|---|---|---|---|---|")
for (v, name), a in acc.items():
    f = lambda k: np.nanmean(a[k])
    print(f"| {v} | {name} | {f('cover'):.2f} | {f('iou'):.2f} | {f('sad'):.2f} | {f('gapA'):.2f} | {f('leak'):.3f} | "
          f"{f('sad_dark_shadow'):.2f} | {f('sad_light_skin'):.2f} |")

"""Scenes and scores for mr_model_eval (model_eval.cpp): the thin-hair research's synthetic scenes and metrics
(ofx/ml/research/vitmatte_thin_hair/exp/synth.py, imported from the folder given).

  python eval_scenes.py gen   <research exp dir> <scene dir>          -> scene_<i>.bin + scene_<i>.npz (truth)
  python eval_scenes.py score <research exp dir> <scene dir> <out dir> -> tables (all scenes, per look)
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, sys.argv[2])
import synth  # noqa: E402

cmd, scene_dir = sys.argv[1], sys.argv[3]

if cmd == "gen":
    os.makedirs(scene_dir, exist_ok=True)
    for sid, (look, w, g) in enumerate(synth.scenes()):
        sc = synth.make_scene(look, w, g, seed=1000 + sid)  # the research's seeds
        u, _ = synth.focus_region(sc["locks"], "circles", 12)
        rgb = synth.encode(sc["img"], "g24").astype(np.float32)
        with open(os.path.join(scene_dir, f"scene_{sid:02d}.bin"), "wb") as fh:
            np.array([synth.C], np.int32).tofile(fh)
            rgb.tofile(fh)
            sc["mass"].astype(np.float32).tofile(fh)
            u.astype(np.float32).tofile(fh)
        np.savez_compressed(os.path.join(scene_dir, f"scene_{sid:02d}.npz"), alpha=sc["alpha"], strands=sc["strands"],
                            mass=sc["mass"], fg_core=sc["fg_core"], scene=np.array([sid, w, g]), look=np.array(look),
                            sep=np.array(sc["sep"]), u_circles_12=u, lock0=sc["locks"][0], lock1=sc["locks"][1])
elif cmd == "score":
    out = sys.argv[4]
    acc = defaultdict(lambda: defaultdict(list))
    for fn in sorted(os.listdir(scene_dir)):
        if not fn.endswith(".npz"):
            continue
        m = dict(np.load(os.path.join(scene_dir, fn)))
        gap, near = synth.zone_masks(m)
        u = m["u_circles_12"] & ~m["fg_core"]
        look = str(m["look"])
        d = os.path.join(out, fn[:-4])
        for p in sorted(os.listdir(d)):
            if not p.startswith("pred_"):
                continue
            v = p[5:-4]
            pred = np.fromfile(os.path.join(d, p), np.float32).reshape(synth.C, synth.C).astype(np.float64)
            r = synth.metrics(pred, m, u, gap, near)
            for k in ("cover", "iou", "sad", "gapA", "sep_ok", "leak"):
                acc[v][k].append(r[k])
                acc[v][k + "|" + look].append(r[k])
    keys = ("cover", "iou", "sad", "gapA", "sep_ok", "leak")
    print("| variant | cover | IoU | SAD/100 | gap alpha | gaps open | leak |")
    print("|---|---|---|---|---|---|---|")
    for v, a in acc.items():
        print(f"| {v} | " + " | ".join(f"{np.nanmean(a[k]):.3f}" for k in keys) + " |")
    print("\nper look: cover / IoU / SAD/100 / gaps open / leak")
    for look in synth.LOOKS:
        for v, a in acc.items():
            f = lambda k: np.nanmean(a[k + "|" + look])  # noqa: E731
            print(f"| {look} | {v} | {f('cover'):.2f} / {f('iou'):.2f} / {f('sad'):.2f} / {f('sep_ok'):.2f} / {f('leak'):.3f} |")

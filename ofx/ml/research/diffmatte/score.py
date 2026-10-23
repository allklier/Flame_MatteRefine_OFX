# Scores the dumps of run_quality.sh: band error (as mr_ml_bench: |sd| <= band around the input
# matte's 0.5 contour, recomputed here as the trimap's unknown region) and strand recovery.
#   lost      L = sum(truth) over pixels where the input matte == 0 (strands the matte lost)
#   recovered R = sum(min(out, truth)) over the same pixels      -> R / L
#   spill     S = sum(max(out - truth, 0)) over the same pixels  -> S / L (alpha added where none was)
# "band" columns restrict the same sums to the trimap's unknown band.
import sys, os, json, glob, numpy as np
Q = sys.argv[1]; W, H = 1920, 1080
rows = []
for d in sorted(glob.glob(f"{Q}/*/*_b*")):
    name, run = d.split("/")[-2], d.split("/")[-1]
    sc, band = run.rsplit("_b", 1)
    f = lambda k: np.fromfile(glob.glob(f"{d}/*_{k}.f32")[0], np.float32)
    try:
        inp, ml, tru, tri = f("in"), f("ml"), f("truth"), f("trimap")
    except IndexError:
        continue
    unk = tri == 0.5
    lost = inp == 0
    def strand(mask):
        L = tru[mask].sum(); R = np.minimum(ml, tru)[mask].sum(); S = np.maximum(ml - tru, 0)[mask].sum()
        R0 = np.minimum(inp, tru)[mask].sum()
        return L, R / L if L else 0, S / L if L else 0
    L, rr, ss = strand(lost); Lb, rrb, ssb = strand(lost & unk)
    rows.append(dict(model=name, scene=sc, band=int(band), in_err=float(np.abs(inp - tru)[unk].mean()),
                     ml_err=float(np.abs(ml - tru)[unk].mean()), lost=float(L), rec=float(rr), spill=float(ss),
                     lost_band=float(Lb), rec_band=float(rrb), spill_band=float(ssb)))
json.dump(rows, open(f"{Q}/scores.json", "w"), indent=1)
order = ["vitmatte-small", "vitmatte-base", "dm-vits-s1", "dm-vits-s3", "dm-vits-s5", "dm-vits-s10",
         "dm-vits-s10-seed1", "dm-vitb-s1", "dm-vitb-s5", "dm-vitb-s10", "dm-vits1024-s10"]
for sc, band in sorted({(r["scene"], r["band"]) for r in rows}):
    rs = {r["model"]: r for r in rows if r["scene"] == sc and r["band"] == band}
    any_r = next(iter(rs.values()))
    print(f"\n{sc} band {band}: input band err {any_r['in_err']:.4f}; lost strand alpha {any_r['lost']:.0f} "
          f"(in band {any_r['lost_band']:.0f})")
    print(f"  {'model':20s} {'band err':>8s} {'rec all':>8s} {'spill':>7s} {'rec band':>8s} {'spill b':>7s}")
    for m in order:
        if m in rs:
            r = rs[m]
            print(f"  {m:20s} {r['ml_err']:8.4f} {r['rec']:8.3f} {r['spill']:7.3f} {r['rec_band']:8.3f} {r['spill_band']:7.3f}")

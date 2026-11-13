"""Summaries of results.csv: python analyse.py results.csv"""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("cover", "iou", "sad", "gapA", "sep_ok", "leak", "leak_own", "w", "g"):
        r[k] = float(r[k])
variants = list(dict.fromkeys(r["variant"] for r in rows))
widths = sorted({r["w"] for r in rows})
looks = list(dict.fromkeys(r["look"] for r in rows))


def agg(sel, key):
    v = [r[key] for r in sel if not np.isnan(r[key])]
    return np.mean(v) if v else np.nan


def table(key, by, vals, title, fmt="{:.2f}"):
    print(f"\n## {title} ({key}, mean over the other factors)")
    print("| variant | " + " | ".join(str(v) for v in vals) + " | all |")
    print("|---" * (len(vals) + 2) + "|")
    for var in variants:
        sel = [r for r in rows if r["variant"] == var]
        cells = [fmt.format(agg([r for r in sel if r[by] == v], key)) for v in vals]
        print(f"| {var} | " + " | ".join(cells) + " | " + fmt.format(agg(sel, key)) + " |")


table("cover", "w", widths, "strand coverage by width")
table("iou", "w", widths, "IoU(alpha>0.5) by width")
table("sep_ok", "g", sorted({r["g"] for r in rows}), "gap kept open (share of gap px < 0.5) by gap")
table("gapA", "w", widths, "false alpha in gaps by width")
table("cover", "look", looks, "strand coverage by look")
table("sad", "look", looks, "SAD/100 in hair zone by look")
table("leak", "look", looks, "leak (> 2 px from strands) by look", "{:.3f}")
table("leak_own", "look", looks, "leak in the variant's own region", "{:.3f}")

print("\n## boundary (base): coverage / IoU / sep_ok per look x width (mean over gaps)")
for look in looks:
    cells = []
    for w in widths:
        sel = [r for r in rows if r["variant"] == "base" and r["look"] == look and r["w"] == w]
        cells.append(f"{agg(sel, 'cover'):.2f}/{agg(sel, 'iou'):.2f}/{agg(sel, 'sep_ok'):.2f}")
    print(f"| {look} | " + " | ".join(cells) + " |")

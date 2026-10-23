# ofx/ml/LICENCES.md — model and runtime licences (moved out of CLAUDE.md 2026-10-23)

Status as of 2026-10-23. Decision 31 (ViTMatte behind the licence gate) and the plan are in `CLAUDE.md`; the licence gate itself is packaging work (P2 in `PLAN_FEATURES.md`).

## Licences (verify each at its source before shipping; status as of 2026-10-23)

| Model | Code | Weights | Status |
|---|---|---|---|
| BiRefNet | MIT (repo LICENSE read) | MIT per listings | the clear one; its training data's terms NOT checked |
| DiffMatte | README says MIT, but **no LICENSE file** and GitHub reports `license: null`; no copyright holder named | **none stated anywhere** (README, Google Drive, paper, 10 issues; checked 2026-10-23, `research/diffmatte/B2_REPORT.md`); trained on Adobe DIM / Composition-1k like ViTMatte | decision 31's basis does NOT carry over: no LICENSE, no authors' statement, no vendor precedent found. Do not redistribute weights or ONNX files. Ask the authors first. |
| ViTMatte | **MIT, verified** (LICENSE file: "Copyright (c) 2023 Hust Vision Lab") | HF card `hustvl/vitmatte-small-composition-1k`: tag Apache-2.0, "trained on Composition-1k", no restrictions stated. **Composition-1k = Adobe's Deep Image Matting dataset (CVPR 2017).** Secondary sources (LibreViTMattes card, which converts this checkpoint; FBA Matting repo) say Adobe's licence agreement allows models trained on it to be used and distributed only non-commercially, not sold, with the CVPR 2017 attribution kept; they say it binds the checkpoint, not the MIT code or independently trained weights | **UNVERIFIED at the source: Adobe's agreement text is not public** (the dataset is on request, contact in the dataset repos); whether "use" covers output on paid work is unknown. Backbone and its pretraining data not named on the pages read |
| ONNX Runtime | MIT | n/a | UNVERIFIED: read the shipped licence |

**Precedent and the authors' word (searched 2026-10-23; vendors not contacted):**
- ONYX Ai Matte (ONYX; paid OFX plugin, lifetime licences €80 / €180 / €450; Resolve, Fusion, Nuke, "any
  OFX app") "uses Meta SAM3 and VitMatte models, packaged via ONNX in FP16" (Digital Production
  2026-03-27); it ships the model in the product and is the closest existing competitor to B1.
- Quick Matte 2 (aescripts) has a "tiling mode with ViTMatte" (from a search excerpt; the page itself
  blocked fetching). Free wrappers: ViTMatte-for-Nuke, MattApp, MatteDesk, ComfyUI nodes.
- **Baselight 7 (FilmLight), confirmed from the installed files (lbm, 7.0.3.26240, read-only listing,
  2026-10-23):** `/Library/Application Support/FilmLight/flexi/effects/uk.ltd.filmlight.vit_matte.v2/`
  (version 2.0.0.25329; the Flexi Effects v2 package, a separate download from filmlight.ltd.uk/flexi,
  not part of the app bundle). Its `.flexi` manifest declares two effects, "ViT Matte" / "ViT Matte High"
  (description "ViT Matte Refine", `output_type: trimap`), each launching `./bin/python3 flexi_vit_matte.py`
  with `VITMATTE_QUALITY=medium|high` and `PYTORCH_ENABLE_MPS_FALLBACK=1` on a PyTorch 2.8 platform: **an
  out-of-process Python worker**, not inline. Weights `ViTMatte_S_Com.pth` (103,348,349 bytes) and
  `ViTMatte_B_Com.pth` (386,835,645 bytes): the authors' checkpoint names and sizes ("Com" = Composition-1k;
  the two quality levels very likely = S and B, an inference), code in the hustvl layout (`modeling/{backbone,
  criterion,decoder,meta_arch}`) plus detectron2, timm 0.5.4, fairscale, fvcore. Its `LICENCE.md` is MIT
  "Copyright (c) 2023 Hust Vision Lab" followed by the Apache 2.0 text (probably detectron2); **no Adobe /
  Composition-1k / non-commercial notice anywhere in it.** FilmLight's other flexi effects: SAM 2.1 (Hiera
  large, Apache 2.0), Depth Anything v2, MiDaS, RIFE retime, face detect/embed/UV. Weights were not
  opened or compared byte for byte with the official files.
- hustvl/ViTMatte issue #9 (Jul 2023, closed): a member answers "Our repo is totally under the MIT
  license. Feel free to use our model weights as long as you follow MIT license." An Oct 2025 comment
  argues the weights cannot be MIT-compatible (COCO CC-NC photos, ImageNet research terms, MAE CC-BY-NC);
  no maintainer reply. Nobody addresses Adobe's Composition-1k terms. Precedent lowers the practical
  risk; it is not clearance (vendors' own arrangements are unknown).

**Decision 31 (DECIDED, user, 2026-10-23): ViTMatte may be used in Matte Refine ML, behind the licence
gate, and is NOT flagged as non-commercial.** Basis: the authors' written statement that the weights are
MIT (hustvl issue #9), and precedent: FilmLight ships the official checkpoints in Baselight 7 and ONYX
sells them in Ai Matte, both with the MIT notice and no dataset notice. It rests on that precedent, not
on a clearance: Adobe's Composition-1k terms were never read at the source and the counter-claims in
the issue are unanswered (the maintainer accepted this; revisit if Adobe, the authors or a vendor ever
says otherwise; not legal advice). Rules that stay: **weights are never in the repo or any release**
(the setup step downloads them for the user, as LTX's does); the gate shows the model's licence
(MIT, "Copyright (c) 2023 Hust Vision Lab"), its source and the Deep Image Matting attribution; the
model file stays swappable. The gate is for non-commercial models only; ViTMatte is not one for us.

Anything non-commercial can only be a user-downloaded model behind a licence gate (LTX's
approach), never bundled. The MatAnyone 2 quality evaluator (locates defects without ground
truth) is on the watch list only: checkpoint unreleased, non-commercial.


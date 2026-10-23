# B2 spike: DiffMatte as ONNX through the Matte Refine ML runner (2026-10-23)

Agent: claude-opus-5-5. Machine: BFD (RTX PRO 4500 Blackwell 32 GB, ORT 1.31.0 cuda13 package, CUDA 13.4
libraries from phase 0). Scratch only: `WORKDIR=.../scratchpad/agents/mlb2/ml_work_b2` (= `$B` below). Nothing
ran in Flame; no repo file edited.

## Verdict: CONDITIONAL GO, reframed. Use DiffMatte as a better 1-step model, not as a slow diffusion "Final"

- **The export works.** It is one ONNX graph per (variant, steps): the ViT encoder runs once and the DUNet decoder
  is unrolled. It uses standard ops only, takes the same `[1,4,512,512]` input as ViTMatte and needs no runner or
  protocol change. ORT matches PyTorch to <= 7.5e-5 (CPU); CUDA vs CPU max 2.4e-3.
- **The diffusion steps add nothing measurable.** On every scene, 1 step equals 10 steps to within 0.0005 band
  error. A different noise seed moves the result more than the steps do. The paper agrees: on AIM-500 (real
  photos) Adobe-trained ViT-B gets worse from 1 to 10 steps (SAD 22.32 to 23.57). A "Final = 10 steps" switch
  would cost 2.8-3.7x the time for no gain.
- **DiffMatte ViT-B at 1 step beats ViTMatte-base on 6 of 7 scene/band rows (worse on hair_busy band 30) at the same
  cost:** HD 181 vs 178 ms coarse,
  161 vs 159 ms hair. The 1024-trained ViT-S is best on busy hair but worst on coarse.
- **"Finer hair detail" is not delivered by any model here.** The synthetic strands lie 60-240 px outside a 10 px
  (or 30 px) unknown band, and no model recovers them (crop: `$B/out/crop_hair_busy_b10.png`). The trimap rule is
  the bottleneck, not the model.
- **The GO depends on the licence.** The weights have no licence at all, and decision 31's basis does not carry
  over (see 1c).

## 1. What DiffMatte is

### 1a. Source
- Repo: github.com/YihanHu-2022/DiffMatte, commit `12082fdbad2ccc319e411ac4a4f3d208780f2961` (2024-07-04).
- Paper: "Diffusion for Natural Image Matting", arXiv 2312.05915 (v2 2024-12-04; the venue is not stated on the
  arXiv page).
- Weights: Google Drive links in the README. Downloaded with gdown:

| Variant | Drive id | Bytes | sha256 | Params |
|---|---|---|---|---|
| ViTS | 11m6y4LI_Hp5u5AagUY_bxPJWkbtsydFV | 348542284 | 057f0dbd27b28537de4ae393a4d8e0dc91e930245db36ec0f17a26d86b1d1cb4 | 29017136 |
| ViTB | 1-Qrtg7txDt53NIreq3PTvgJke6UhuB2z | 1217878540 | b0a373a7131593b184f0ca4f6e36bac09871a6955748f5f7a331b61bbb71b892 | 101461808 |
| ViTS(1024) | 1NIn-tKtW3zhi2vK3OgOTiiHrOIXuHIZo | 348935500 | 685fdd14fd07a26be1075b8a4c2b4175e448a298f9d92bb857f54e3fb8636029 | 29049904 |

- The checkpoints are detectron2 training checkpoints and include the optimizer state, which is why they are about
  3.5x the model size.
- Their pickles reference omegaconf data classes. I checked every global with pickletools (OrderedDict, tensors,
  omegaconf `ListConfig` / `ContainerMetadata` / `Metadata` / `AnyNode`, `typing.Any`, builtins), then loaded them
  with `weights_only=True` with exactly those allowlisted (`scripts/clean_ckpt.py`).
- The parameter counts match the README (29.0 M / 101.4 M).

### 1b. How it works (read in `modeling/meta_arch/difmodel.py`, `difmatte.py`, `diffusion/uniform_gauss.py`, `decoder/DUNet.py`)
- **Encoder:** the ViTMatte ViT (ViTDet-style, windowed attention plus 4 global blocks, decomposed relative position
  terms, residual conv blocks). Input is 4 channels: RGB 0-1 ImageNet mean/std normalised, plus the trimap
  0/0.5/1. Output features are [1,384 or 768,T/16,T/16]. **It runs once.**
- **Decoder (DUNet, iterated):** a small conv UNet (Conv, ConvTranspose, bilinear upsample, BatchNorm, timestep
  MLP). Full-resolution input of 7 channels: normalised RGB plus the one-hot trimap, mapped `*2-1`, plus x_t
  divided by its own global std (scale 0.2). It predicts x0 (sigmoid*2-1).
- **Sampler:** `UniformGauss.ddim_sample`, linear gamma schedule, times linspace(1,0,steps+1), scale 0.2, initial
  x_t = `torch.randn`. **Non-deterministic as shipped** (repo issue #3, "Different output each time").
- **Recommended steps:** the README reports "step 10"; `run_one_image.py` defaults to `--sample-strategy ddim10`;
  the configs say `diffusion.steps = 1` (overridden at inference). The paper reports S1 and S10 and says results
  "improve as the number of iteration steps increases", except ViT on D646 and ViT-B on AIM-500.
- **Paper cost at 2048²:** ViT-S first step 578 ms, then 126 ms per step.
- **Output:** clamp 0-1, and trimap==0 is forced to 0. Their "trimap==2 -> 1" never fires because the trimap is
  0/0.5/1.

### 1c. Licence (verified at the source unless marked)
- **Code: MIT is claimed, but the repo has no LICENSE file.**
  - Verified: no `LICEN*` or `COPYING` file at the commit above.
  - Verified: GitHub API `license: null`.
  - README, quoted: "The code is released under the MIT License. It is a short, permissive software license.
    Basically, you can do whatever you want as long as you include the original copyright and license notice in
    any copy of the software/source."
  - The README also has an MIT badge. No copyright holder is named anywhere.
  - `index.html` (project page) is CC BY-SA 4.0.
- **Weights: no licence stated anywhere.** Not in the README, not on Drive (only the links), not in the paper. The
  arXiv paper carries only "arXiv.org perpetual non-exclusive license" (for the paper) and "The code will be
  open-sourced for the following research and applications."
- **Training data:**
  - README, quoted: "Get DIM dataset on Deep Image Matting" and "For Training, merge 'Adobe-licensed images' and
    'Other' folder to use all 431 foregrounds and alphas", with backgrounds from PASCAL and COCO.
  - Paper: "The training set is constructed by compositing each foreground with 100 background images from the
    COCO dataset."
  - So these are **Composition-1k (Adobe DIM) checkpoints, exactly like ViTMatte's.** The paper also trained ViT
    variants on Distinctions-646; those weights are not published (issue #7 asks for them).
- **Backbone init: UNVERIFIED.** The paper does not say. `pretrained/preprocess.py` defaults to `name='dino'`, which
  hints at DINO ImageNet weights (an inference).
- **For decision 31:** ViTMatte was cleared on (a) a LICENSE file, (b) the authors' written statement that the
  weights are MIT (hustvl issue #9), and (c) precedent (FilmLight, ONYX). For DiffMatte, (a) is missing, (b) does
  not exist (no issue or statement found among its 10 issues) and (c) was not found (not searched beyond the repo).
  **Treat it as unlicensed weights plus Adobe DIM terms.** A redistributable ONNX file is out; even a user-download
  needs the maintainer's licence decision. Asking the authors for a LICENSE file and a statement on the weights
  (as hustvl gave) would be the clean path.

## 2. ONNX export: done, no blockers

- **Wrapper `scripts/dm_common.py: Tile`.**
  - Input: `pixel_values` [1,4,T,T], **exactly the ViTMatte prep in `edge_refine.cpp` `prepareTile`**: RGB
    (x-0.5)/0.5, trimap 0/0.5/1. Inside the graph it becomes ImageNet normalisation, a one-hot trimap (comparisons,
    exact for 0/0.5/1) and the DDIM loop unrolled with constant timesteps.
  - The initial noise is a **constant in the graph** (seed 0, 1 MB at 512), so the output is deterministic and
    stable over time for the same input.
  - Output: `alphas` [1,1,T,T], clamped, trimap==0 set to 0.
  - The decoder weights are shared across the unrolled steps: 10 steps add only 0.2 MB.
- **Fidelity:**
  - Wrapper vs the authors' `ddim_sample` with the same noise: max 6.9e-7 (demo image, 1 and 10 steps).
  - ORT 1.31 CPU vs PyTorch: max 1.8e-5 to 7.5e-5, mean ~1e-7, on every export.
  - CUDA EP vs CPU EP on a whole hair frame: max 2.4e-3, mean 1.2e-6.
  - The demo image (retriever) gives a correct fur matte (`$B/out/demo/demo_strip.png`).
- **Ops:** Add, And, BatchNormalization, Cast, Clip, Concat, Conv, ConvTranspose, Div, **Einsum** (rel-pos, as in
  ViTMatte), Erf, Gather, Gemm, LayerNormalization, MatMul, Pad, Resize, Softmax, etc., opset 17. No custom ops,
  no deformable attention. `mr_ml_bench` loads them unchanged.
- **Shapes: fixed 512 x 512.** The ViT code takes Python branches on shapes at trace time (`get_abs_pos`
  interpolation, window padding), so a dynamic-shape trace would be wrong at other sizes. I did not try making it
  dynamic. Export one file per tile size instead (512 is phase 0's best size). ViTS(1024) at tile 512 works: its
  pos-embed resize is folded into the graph.
- **Split encoder / step-decoder:** possible (DifModel already accepts `features=`), but it needs multi-input RUNs
  and gives nothing, because 1 step is enough. **Not recommended.**

| Model (tile 512) | Bytes | sha256 |
|---|---|---|
| diffmatte-vits-s1 | 117654245 | 9fb152af01f87d80d55d70a582ae53e647e1a61137c09eae607f9b3d3feca0d3 |
| diffmatte-vits-s3 | 117706279 | 346f311a8d5537cec43efe859975c68b1ebd7cca1372d328c403ac5c57d61b24 |
| diffmatte-vits-s5 | 117758337 | 2ff4aae4b3feb07387549cada58d24cfdd52c96288b88854af07d44f125b689f |
| diffmatte-vits-s10 | 117888840 | 40512b1b973b9529383bf44cd0c2387d2cc942f2ffdbe1441cb283b931da0e72 |
| diffmatte-vits-s10-seed1 | 117888840 | 2e6bb02ac917ed0053f13124f7db6d6b0ecae3d0a6a0db291c4f1bea94d59e78 |
| diffmatte-vitb-s1 | 407431456 | 2f1065a53d0895bc156b4067a71a59543cea23c8044f956986b32f1071b2ae5e |
| diffmatte-vitb-s5 | 407535548 | ed7002467ea865de74eb1e2fc0830060acbd4d2947035a0ea2f277c8a44e7f82 |
| diffmatte-vitb-s10 | 407666051 | 7a1dc68d36ce132f76992b03dd52f0ea7c394ffd0d81f6781350f58f7eb2903f |
| diffmatte-vits1024-s1 | 117791675 | cbf6dc30b57b7c5e20468e8ab0e97f407b86fb84b847bf74f5472578a8ac3ed3 |
| diffmatte-vits1024-s10 | 118026270 | ddd6703cb3a54a21ab433a407e8a3e38e65ab9a5098c86b86f822fa2d28ebcbc |

(These are `$B/models/<name>/model.onnx`. They contain the unlicensed weights: never commit or redistribute them.)

## 3. Speed and memory (CUDA EP, tile 512, band 10 px, warm median of 5)

- **Contention while measuring:** while the quality runs were going, other agents' processes (B3 `eval_smart_fill`
  / `eval_bin` 29.4 GB, an `mr_ml_runner` 1.5 GB) filled the GPU. That caused OOMs, so 20 quality runs were done on
  the CPU EP instead. Timing was run later with the GPU idle (0 %, 1 GB used). A few timing runs overlapped brief
  other users of up to 1.4 GB (printed per run to the console; the text log was lost, the numbers are in `$B/out/timing/timing.jsonl` and `timing_summary.txt`): dm-vits-s10
  hair/UHD and dm-vitb-s1. Their medians agree with the per-tile rate; their p95 values were discarded.
- **Today's numbers vs phase 0:** today's ViTMatte-small tile is 11.7 ms against 8.2 ms in phase 0 (another GPU
  state). Compare within this table.

| Model | per 512 tile | HD coarse (9 tiles) | HD all (12) | HD hair_busy (8) | UHD all (29) | load / first frame | GPU delta |
|---|---|---|---|---|---|---|---|
| ViTMatte-small | 11.7 ms | 114 ms | 143 | 95 | 370 | 0.8 s / 0.3 s | +0.9-1.4 GB |
| ViTMatte-base | 18.5 | 178 | 235 | 159 | 578 | 1.1 / 0.36 | +1.4 |
| DM ViT-S s1 | 11.7 | 121 | 161 | 107 | 394 | 0.8 / 0.36 | +1.4 |
| DM ViT-S s3 | 19.7 | 191 | 246 | 165 | 616 | 0.9 / 0.43 | +1.4 |
| DM ViT-S s5 | 28.2 | 267 | 346 | 232 | 849 | 1.0 / 0.55 | +1.4 |
| DM ViT-S s10 | 47.3 | 442 | 592 | 393 | 1626 | 1.1 / 0.7 | +1.4-2.2 |
| DM ViT-B s1 | 18.9 | 181 | 269* | 161 | 601 | 1.1 / 0.39 | ~+2.5 |
| DM ViT-B s5 | 34.5 | 323 | 434 | 290 | 1052 | 1.2 / 0.86 | +2.1-2.5 |
| DM ViT-B s10 | 55.1 | 506 | 678 | 452 | 1646 | 1.25 / 0.73 | +2.5 |

- `*` = overlapped another process.
- **Decoder cost: about 4.0 ms per step per 512 tile** for both S and B. Encoder: about 7.7 ms (S) / 15 ms (B).
- ViTS(1024) has the same architecture as ViT-S: same cost (not timed separately).
- RSS peak 1.3-1.7 GB. CPU EP: 0.29 s (S s1) to 1.9 s (B s10) per tile, so CPU is not a path.
- **Against decision 28's gate (~1 s warm HD, never 3 s):**
  - s1 models are inside it like ViTMatte.
  - 10 steps are inside it at HD (0.4-0.7 s) and reach 1.6 s at UHD.
  - Under a Batch Render (+45 %, phase 0) that becomes about 2.4 s at UHD.

## 4. Quality (1920x1080, trimap and blend exactly as `edge_refine` / `mr_ml_bench`)

- **Band error:** mean |alpha - truth| over the trimap unknown band.
- **Strand recovery:** over pixels where the input matte == 0, R = sum(min(out, truth)) / sum(truth). "In band"
  restricts this to the unknown band.
- `scripts/score.py`, full table: `$B/out/quality/scores.txt` and `scores.json`.

| Band error | input | ViTM-S | ViTM-B | DM-S s1 | DM-S s10 | DM-S s10 seed1 | DM-B s1 | DM-B s10 | DM-S1024 s1 | DM-S1024 s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hair_flat b10 | 0.189 | 0.130 | 0.104 | 0.097 | 0.096 | 0.096 | **0.073** | 0.073 | 0.097 | 0.098 |
| hair_busy b10 | 0.189 | 0.107 | 0.091 | 0.092 | 0.091 | 0.091 | 0.077 | 0.077 | **0.053** | 0.053 |
| hair_flat b30 | 0.104 | 0.089 | 0.086 | 0.081 | 0.082 | 0.082 | **0.072** | 0.072 | 0.077 | 0.077 |
| hair_busy b30 | 0.104 | 0.081 | 0.063 | 0.081 | 0.082 | 0.082 | 0.069 | 0.070 | **0.054** | 0.054 |
| coarse b10 | 0.221 | 0.032 | 0.031 | 0.033 | 0.032 | 0.034 | **0.029** | 0.029 | 0.038 | 0.038 |
| eroded b10 | 0.118 | 0.0082 | 0.0080 | 0.0104 | 0.0097 | 0.0098 | **0.0055** | 0.0052 | 0.0089 | 0.0088 |
| mixed_edge b10 | 0.021 | 0.0086 | 0.0080 | 0.0122 | 0.0116 | 0.0118 | **0.0060** | 0.0059 | 0.0104 | 0.0102 |

- **Hair strand recovery** (hair_busy band 10; lost strand alpha 28446, of which 6590 lies in the band):
  - In the band: ViTM-S 0.59, ViTM-B 0.61, DM-S 0.60, DM-B 0.60, DM-S1024 0.78.
  - Whole frame: 0.137 / 0.140 / 0.140 / 0.140 / 0.181. Spill is 0.4-3 % of the lost alpha.
  - hair_flat: everyone 0.56-0.62 in the band and 0.13-0.14 in the frame.
  - Band 30: 0.31-0.53 in the band, 0.13-0.22 in the frame.
- **About 80 % of the lost strand alpha is outside even a 30 px band, so no model can touch it.** The crop shows all
  models only clean the head contour; S1024 adds some strand roots.
- **Reading:**
  - DM ViT-B s1 is the best or equal-best model on 5 of the 7 rows. Its band error vs ViTMatte-base is -30 % hair_flat b10, -16 % hair_busy
    b10, -16 % hair_flat b30, +10 % (worse) hair_busy b30, -6 % coarse, -31 % eroded, -25 % mixed_edge.
  - DM ViT-S is about ViTMatte-base on hair and worse on eroded / mixed.
  - DM S1024 wins clearly on the busy background (0.053 vs 0.077-0.107) but loses on coarse.
  - Steps 1 to 10 change nothing beyond about 0.0005.
- **Caveats:**
  - The synthetic hair is much simpler than real hair: flat strands, a known composite. The picture contains the
    true edge (as noted in phase 0).
  - The clean-matte (fgcolor) row was not rerun.
  - Real footage decides.

## 5. Integration plan, if the licence is cleared

1. **Model table:**
   - Rows `diffmatte-vitb` (1 step) and optionally `diffmatte-vits1024` (1 step): channelsIn 4, channelsOut 1.
   - `kMultipleOf` with multiple 512, minSide = maxSide = 512 (the export is fixed); tensor names `pixel_values` /
     `alphas`.
   - Licence id e.g. `diffmatte-dim`, sha256 per file.
   - No runner, protocol or slot change: slot = 4+1 planes x 512² x 4 B = 5 MiB < 20 MiB.
2. **edge_refine:**
   - The same `prepareTile` / `accumulateTile` / `blendBack`.
   - A frame smaller than 512 must pad to 512 (today the tile shrinks to fit: a few lines, or skip the model).
3. **Plugin UI:**
   - A **Model** choice inside Edge Refine (ViTMatte Small / Base / DiffMatte B / DiffMatte S-1024).
   - **No Steps or Quality = steps control:** it costs time and changes nothing measurable.
   - If a "Final" is wanted, make it = DiffMatte-B s1; Draft = ViTMatte-small.
4. **Never random noise per frame:** it is a constant inside the graph. The noise changes results by about 0.002
   (seed 0 vs 1), so it is part of the model file and is never re-drawn.
5. **Setup:**
   - The weights cannot be rehosted. The setup must download the `.pth` from Google Drive (gdown-style confirm
     handling; Drive quotas are a reliability risk) and convert it on the user's machine.
   - Either ship a weightless ONNX (external-data) plus a small restricted-unpickler converter (numpy only, the
     allowlist above), or run torch + detectron2 locally (heavy; detectron2 built from source here).
6. **Effort:**
   - Model rows, pad rule, UI choice and tests: about 0.5-1 day.
   - Weightless-graph + converter setup path: about 1-1.5 days.
   - The licence gate entry is part of P2.
   - In total about 2-3 days, plus the licence decision and a real-footage comparison in Flame.
7. **Before integrating:**
   - Fix the trimap for hair (B1 polish: a wider or faint-alpha-aware unknown region). Without it, no model choice
     recovers strands beyond the band.
   - Test DM-S1024 at tile 1024 (not done here; trained at 1024 crops).

## Reproduce

Paths are given with `B=<WORKDIR>/ml_work_b2` and `S=.../scratchpad/agents`.

```
# environment (CPU torch; detectron2 from source)
python3 -m venv $B/venv && $B/venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
$B/venv/bin/pip install timm==0.5.4 fvcore fairscale einops tqdm opencv-python-headless onnx onnxscript gdown numpy onnxruntime==1.31.0 wheel setuptools
git clone https://github.com/facebookresearch/detectron2 $B/dl_d2/detectron2 && $B/venv/bin/pip install --no-build-isolation $B/dl_d2/detectron2
git clone https://github.com/YihanHu-2022/DiffMatte $B/dl_repo/DiffMatte; (cd $B/dl_repo/DiffMatte && find modeling -name '*.py' | cpio -pdm $B/dm_src)
# weights (each in its own dir), then clean state dicts
cd $B/dl_w_vits && $B/venv/bin/gdown 11m6y4LI_Hp5u5AagUY_bxPJWkbtsydFV -O ckpt.pth   # vitb: 1-Qrtg7..., vits1024: 1NIn-...
cd / && $B/venv/bin/python -I $B/scripts/clean_ckpt.py $B/dl_w_vits/ckpt.pth $B/weights/vits.pt
# reference check and export (variant, weights, steps, tile, out[, seed])
$B/venv/bin/python -I $B/scripts/check_ref.py vits $B/weights/vits.pt $B/dl_demo/retriever_rgb.png $B/dl_demo/retriever_trimap.png $B/out/demo
$B/venv/bin/python -I $B/scripts/export_onnx.py vitb $B/weights/vitb.pt 1 512 $B/models/diffmatte-vitb-s1/model.onnx
# bench (repo copy, out of tree)
cmake -S <WORKDIR> -B $B/build -DMR_BUILD_ML=ON -DORT_DIR=$S/mlbench/ml_work/ort/onnxruntime-linux-x64-gpu_cuda13-1.31.0 -DCMAKE_BUILD_TYPE=Release
cmake --build $B/build -j16 --target mr_ml_bench
CUDA_LIBS=$S/mlbench/ml_work/cuda_libs [EP=cpu] $B/scripts/run_quality.sh $B $S/mlbench/ml_work/models
$B/venv/bin/python -I $B/scripts/score.py $B/out/quality
CUDA_LIBS=$S/mlbench/ml_work/cuda_libs $B/scripts/run_timing.sh $B $S/mlbench/ml_work/models   # GPU idle
$B/venv/bin/python -I $B/scripts/crops.py $B/out/quality hair_busy_b10 $B/out/crop_hair_busy_b10.png
```

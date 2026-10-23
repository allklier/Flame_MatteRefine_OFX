# ofx/ml/CLAUDE.md — Matte Refine ML (plan; nothing built yet)

The second plugin: ML and slow matte refinement, in this repo and CMake project next to Matte
Refine (root decisions 17, 18, 27). Plugin 1 stays real-time and inline (14). Goal, architecture and
the decision index are in the root `CLAUDE.md`; the operations plugin 2 reuses are in
`ofx/src/ops/CLAUDE.md`; Flame host rules in `ofx/CLAUDE.md`. Tags as there (DECIDED / PROPOSED /
OPEN / UNVERIFIED). Written 2026-10-23. Keep this file under ~250 lines.

## Scope (decision 17 holds)

The input is a good matte (LTX or elsewhere); every ML operation **keeps it and fixes defects**.
Whole-matting models (MatAnyone, VideoMaMa, GVM, SAM2Matting, SDMatte, keyers) stay out. The
user's picture (front input) is the guide, the matte (Matte input) is the subject; same inputs
and colour rules as plugin 1 (decisions 22-24).

| # | Operation | What it does | Model (licence, see below) |
|---|---|---|---|
| B1 | **Edge Refine** | trimap from the matte itself (solid core / grey band widened by N px / clear); the model re-solves only the band; Mix + "only where the matte was grey"; outside the band bit-identical | ViTMatte (first), others later |
| B2 | **Edge Refine, diffusion** | same trimap, finer hair detail, slower | DiffMatte |
| B3 | **Smart Fill Holes** | a segmentation says what is the subject: holes inside it are defects, holes outside are real gaps (replaces the Max Hole % guess) | BiRefNet |
| B4 | **Temporal Stabilise** | neighbour frames, motion-compensated, only the edge band and Fill Holes decisions; classical, the matte stays the source | none (optical flow) |

## Decisions

- **27. DECIDED (user, 2026-10-23): one repo, one CMake project, two bundles.** `ofx/ml/` holds
  the plugin; a second `add_library(MatteRefineML MODULE)` target next to `MatteRefine`,
  bundle `MatteRefineML.ofx.bundle`, plugin id `com.janklier.matterefineml`, its own version
  and release scripts. Both link the shared static libraries (`mr_ops`, edge geometry, log,
  platform, `mr_synth_lib`). Why not one bundle with two plugins: Flame loads a bundle as one
  binary, so a heavy runtime (GPU libraries) in it could stop the light plugin from loading, and
  the ML one should be an optional, large install. CMake option `MR_BUILD_ML` (default OFF: a
  plain build stays light and needs no runtime). The 1.4 OFX support library is reused; the
  colour headers (23) are already vendored.
- **28. DECIDED (user, 2026-10-23): models run inline in `render()` first; an offline worker only
  if measured performance forces it.** Inline = Matte Refine's model (14): scrub, tweak, Batch
  Render just works. If inline locks Flame up too much or is too slow, split like LTX Alpha
  Gen: a resident worker process (`~/ClaudeCode/LTX_OFX`, its `CLAUDE.md` and `worker/`), Process
  button, disk cache (decision 14 explicitly does not apply to plugin 2 then). **The decision
  is made on measurements** (plan below), in Flame, not on estimates.
- **29. PROPOSED: runtime = ONNX Runtime in the .ofx** (C API; CUDA execution provider on Rocky,
  CoreML or CPU on the Mac), models as `.onnx` files in a model folder fetched by a setup step with
  a licence gate (as LTX's Setup). Why: one shared library instead of libtorch's size; the
  same model file on both platforms; the worker fallback can load the same files. Alternatives
  to weigh at the spike: TensorRT (NVIDIA only, fastest), libtorch (largest, but what LTX
  already ships and verified under Flame's driver), Core ML directly on the Mac. See "Runtime
  facts".
- **30. PROPOSED: band-only, tiled processing.** The trimap comes from the matte with
  `ops/edge_field` (signed distance: core = deeper than N px, unknown = within N px of the 0.5
  contour, clear = beyond); only tiles that touch the unknown band go through the model; the
  result replaces the band and fades to the input at its border (same seam rule as Snap to
  Image). Cost then scales with the edge length, not the frame. Everything outside the band is
  bit-identical to the input (the plugin-1 rule, decision 17).

## Plan, in order (each phase ends with a go / no-go for the next)

0. **Benchmark first, no plugin. DONE on BFD 2026-10-23 (results below); target-host runs open.** A standalone tool `ofx/ml/tools/mr_ml_bench`: loads an ONNX
   file with the candidate runtime and runs tiles cut from the synthetic scenes (21) and the
   user's real mattes (the Save Matte dumps; confidential, stay on the machine) at HD and UHD.
   Measure: warm and cold latency per tile (256, 512, 1024), tiles per typical frame, total
   per-frame time, VRAM, model load time, host RAM; on akflame5 (RTX PRO 6000 Blackwell) and lbm
   (M1 Ultra); and **while Flame is rendering** (GPU contention, the LTX nodes' "unload GPU"
   lesson). Candidates: ViTMatte small and base (fp32, fp16), BiRefNet lite and standard.
   Output: a table here, and the quality of each on the synthetic "coarse" / "eroded" scenes
   against their truth (B1) and on a real frame.
1. **Spike plugin** (inline): `MatteRefineML` with one node, B1 only, ViTMatte, lazy model load,
   per-stage timings in the log, no polish. The Flame test answers what the numbers cannot:
   does the UI freeze while a frame renders, does scrubbing work, Batch Render, memory growth,
   what happens when two nodes run, does Flame render a frame twice.
2. **Decision 28's gate:** stay inline / add a worker with Process + cache / both (inline for
   the light models, worker for DiffMatte). **PROPOSED thresholds** (the user sets them): a warm
   HD frame of a typical subject within about 1 s and no single frame longer than about 3 s
   inline; otherwise a worker. UNVERIFIED: whether Flame caches OFX results between frames
   (Neat-style "middle ground"): test it in phase 1.
3. B3 (BiRefNet guided fill), then B2 (DiffMatte) on the chosen architecture.
4. B4 (temporal): needs neighbour frames (`setTemporalClipAccess`, UNVERIFIED in Flame) and
   probably the cache; last.
5. Packaging: install script, model setup and licence gate, release notes, Mac build.

## Phase 0 results (2026-10-23, built by an Opus agent; `ofx/ml/tools/mr_ml_bench`)

Machine: BFD, i9-12900KS, RTX PRO 4500 Blackwell 32 GB, driver 595, ONNX Runtime 1.31.0 (cuda13
package), CUDA 13.4 libraries from PyPI. **Not yet run on the target hosts** (akflame5 Rocky 9, driver
580.159; lbm M1 Ultra); a run package exists (`fetch.sh`, `build_rocky9.sh`, `build_mac.sh`,
`run_bench.sh`, README; in `ofx/ml/package/`). Warm frames, tiles one after another, trimap band 10 px at HD:

| ViTMatte-small, CUDA EP | tiles | per tile | frame |
|---|---|---|---|
| coarse HD, tile 256 / 512 / 1024 | 17 / 9 / 4 | 3.4 / 8.2 / 44 ms | 78 / 112 / 240 ms |
| all HD, tile 256 / 512 | 31 / 12 | 3.5 / 8.2 ms | 144 / 148 ms |
| coarse UHD, tile 256 / 512 | 49 / 18 | 3.5 / 8.2 ms | 229 / 231 ms |
| all UHD, tile 256 | 85 | 3.5 ms | 395 ms |
| real t36 (2880×1620), tile 512 | 15 | 8.2 ms | 189 ms |
| real t44 (2880×1620), tile 512 | 8 | 8.2 ms | 103 ms |

Cold: model load 380-510 ms, first inference ~200 ms. GPU memory 0.9 → 1.6-3.4 GB, RSS 1.2-1.4 GB.
Base model 1.3× the small one (coarse HD tile 512: 172 ms). CPU EP: 1 s (HD, tile 256) to 4.5 s (UHD).
Pre-processing (single-threaded `pow`) is ~30 % of the frame: easy to speed up. 1024 tiles cost 5×
more per pixel than 512. **BiRefNet** (whole frame, 1024²): lite 100-113 ms, standard 163-220 ms per
inference (fp16 faster), HD frame 150-272 ms; **8.5-16.7 GB of GPU memory** (likely ORT's arena: not
yet limited), load 2-3 s, CPU 4.3 s and 12 GB.

**Quality, mean |error| in the band vs the truth** (input / ViTMatte-small blended back / Snap to
Image): coarse HD 0.221 / **0.033** / 0.095; eroded HD 0.118 / **0.010** / 0.076; all HD 0.079 /
**0.020-0.026** / 0.063; mixed_edge 0.021 / 0.010 / 0.0105 (tie); **clean matte (fgcolor) 0.000 /
0.013 / 0.005 (both ML and Snap make it worse)**. Base is no better than small. Tile size barely
matters. Crops: eroded arm edge matches the truth; on coarse the **hair outside the 10 px band is
removed** (the trimap marks it clear) and a faint grey ring shows inside the blend. BiRefNet "all":
IoU 0.97-0.98, but it closes most real gaps (hole components: input 24, model 1-4, truth 7), so it is
not yet a drop-in inside/outside signal for B3. The synthetic picture contains the true edge, which
flatters both methods: real footage decides.

**Facts settled:** ORT's Linux package needs only glibc 2.28 / GLIBCXX 3.4.22, so **it fits Rocky 9**
(libonnxruntime 33 MB + CUDA provider 284 MB; the CUDA 13 libraries are NOT in the tarball: 1.6 GB
from PyPI wheels, found via LD_LIBRARY_PATH; total ORT+CUDA ~2 GB on disk; MIT licence + third-party
notices). **The ORT 1.31 headers need C++17** (even the C API header): ORT must sit in its own C++17
translation units (the plugin is C++14 for OFX's auto_ptr). The macOS package
(`onnxruntime-osx-arm64-1.31.0.tgz`, 36 MB) contains the CoreML EP (`AppendExecutionProvider("CoreML",
{})`, untested). Models and checksums are in the agent's `models.json`: vitmatte-small ONNX 104 MB,
base 387 MB, birefnet-lite 224 MB (fp16 115 MB), birefnet 973 MB (fp16 490 MB); BiRefNet card:
trained on DIS-TR, licence MIT.

**Reading against the proposed gate (decision 28):** inline looks feasible on this class of GPU:
a warm HD frame 80-150 ms and UHD 200-400 ms are far inside "about 1 s / never 3 s" and similar to
the heavy plugin-1 chains. What the numbers cannot show: contention with Flame's own GPU use, the
CUDA 13.4 runtime against Flame's driver 580 (minor-version compatibility, untested: may need a
13.0/13.2 runtime), blocking Flame's UI for 0.1-0.4 s per frame, first-frame cost (~0.7 s). Those are
the target-host runs and the phase 1 spike.

**Consequences for the design (decision 30):** B1 needs a "keep the input where it was fine" rule:
the ML result hurts a clean matte (Mix, "only where the matte was grey", or run only where Analyze
sees defects); a wider band, or a trimap rule for faint hair, so hair beyond the band survives;
GPU memory limits (`gpu_mem_limit`, arena strategy) before it can share the card with Flame; the
colour conversion into the model (the bench used clamp + gamma 2.2; OPEN).

## Licences (verify each at its source before shipping; status as of 2026-10-23)

| Model | Code | Weights | Status |
|---|---|---|---|
| BiRefNet | MIT (repo LICENSE read) | MIT per listings | the clear one; its training data's terms NOT checked |
| DiffMatte | MIT (repo) | UNVERIFIED; very likely trained on Composition-1k like ViTMatte | check the README / paper |
| ViTMatte | **MIT, verified** (LICENSE file: "Copyright (c) 2023 Hust Vision Lab") | HF card `hustvl/vitmatte-small-composition-1k`: tag Apache-2.0, "trained on Composition-1k", no restrictions stated. **Composition-1k = Adobe's Deep Image Matting dataset (CVPR 2017).** Secondary sources (LibreViTMattes card, which converts this checkpoint; FBA Matting repo) say Adobe's licence agreement allows models trained on it to be used and distributed only non-commercially, not sold, with the CVPR 2017 attribution kept; they say it binds the checkpoint, not the MIT code or independently trained weights | **UNVERIFIED at the source: Adobe's agreement text is not public** (the dataset is on request, contact in the dataset repos); whether "use" covers output on paid work is unknown. Backbone and its pretraining data not named on the pages read |
| ONNX Runtime | MIT | n/a | UNVERIFIED: read the shipped licence |

**Decision 31 (PROPOSED, 2026-10-23): use ViTMatte for evaluation only** (phase 0 and the spike):
no weights in the repo or any release, ever; the plugin fetches models for the user behind a
licence gate that shows the model's terms (LTX's approach). Decide the licence question only if
ML clearly beats Snap to Image in the benchmark: then ask Adobe (or counsel) about commercial use
of a Composition-1k-trained model, or pick a model with clean provenance. Most trimap matting
models (FBA, MGMatting, ViTMatte, DiffMatte probably) share this dataset; BiRefNet does not. The
model file is swappable: nothing in the plugin design depends on ViTMatte. Not legal advice.

Anything non-commercial can only be a user-downloaded model behind a licence gate (LTX's
approach), never bundled. The MatAnyone 2 quality evaluator (locates defects without ground
truth) is on the watch list only: checkpoint unreleased, non-commercial.

## Runtime facts (2026-10-23, from the web; UNVERIFIED unless a line says otherwise)

- ONNX Runtime: CUDA 13 is the default CUDA EP package from 1.27; 1.28 is current; 1.20.1 does
  not work with CUDA 13; Blackwell (sm_120) fixes are in 1.28; INT8 models may lack Blackwell
  kernels (use fp32 or fp16). Flame's shipped NVIDIA driver runs CUDA 13.2 workloads
  (verified in LTX with torch 2.13, driver 580.159.04, `LTX_OFX/CLAUDE.md` decision 22).
  OPEN: the glibc a prebuilt ORT needs (Rocky 9 is glibc 2.34 / GCC 11, root decision 2); the CoreML EP
  on macOS 26 (not found in the search).
- ViTMatte ONNX exists (a Hugging Face conversion); its Einsum ops keep CoreML on the CPU, and
  a rewrite to MatMul (bit-identical on CPU) is reported. ~26 M parameters (small).
- BiRefNet: about 94 ms per 1024² frame (lightweight variant, A100, ONNX, from its README) and
  about 59 ms in fp16 PyTorch on an RTX 4090 (3.5 GB); a browser WebGPU build ~700 ms.
- DiffMatte: the encoder runs once and a light decoder iterates (paper); no timing or
  checkpoint sizes found: read the repo.

## Risks and open questions

- **Flame's main thread**: every OFX action, render included, runs there (`ofx/CLAUDE.md`); a
  slow model blocks the UI. The spike measures exactly this. Tiles can be processed in
  time-sliced chunks only if Flame tolerates a render returning early: it does not (OPEN).
- **GPU memory and contention** with Flame (and LTX nodes on the same machine).
- **First-frame cost**: loading the model (seconds) on the first render after Flame starts.
- **Quality on real footage** is unproven: the synthetic scenes measure geometry, not hair.
  The user's real frames decide whether B1 beats Snap to Image (plugin 1, no ML).
- **Colour**: the picture must be in the colourspace the model expects (a conversion or a
  host request through colour management, root decision 24's check applies).
- OPEN: setup/download location and size budget on akflame5 (`/home` was 94-96 % full);
  Mac install path; whether the ML bundle gets its own Version label / log (the log can be shared).

## Repo layout (when built)

`ofx/ml/src/` (plugin, runtime wrapper, tiling, model registry), `ofx/ml/tests/`,
`ofx/ml/tools/mr_ml_bench`, `ofx/ml/CLAUDE.md` (this), `packaging/` an install script for the ML
bundle, models never in the repo (the setup step downloads them).

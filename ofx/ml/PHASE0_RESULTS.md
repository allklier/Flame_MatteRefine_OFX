# ofx/ml/PHASE0_RESULTS.md: benchmark results for Matte Refine ML (reference data)

Moved out of `ofx/ml/CLAUDE.md` on 2026-10-23 (that file's size limit). Tool: `ofx/ml/tools/mr_ml_bench`,
run package and Flame test instructions: `ofx/ml/package/`. Summary and decisions: `ofx/ml/CLAUDE.md`.

## BFD and akflame5 benchmarks (2026-10-23)

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

**Phase 0 on the target host, akflame5 (2026-10-23, run by the lead from the package, 94 runs):**
Rocky 9.7, Xeon 658X, RTX PRO 6000 Blackwell 97.9 GB, **Flame's driver 580.159.04, Flame not running (GPU
idle: no contention measured yet)**, ONNX Runtime 1.31.0 with the PyPI **CUDA 13.4 libraries: they run
under Flame's driver 580** (the open question is answered: no pin to 13.0/13.2 needed; no ORT warnings
in stderr). Tool built with the system GCC 11 (selfcheck ok). Warm frame median, tiles one after another:

| ViTMatte-small, CUDA | tile 256 | tile 512 | tile 1024 |
|---|---|---|---|
| HD coarse (17 / 9 / 4 tiles) | 90 ms | 98 ms | 172 ms |
| HD all (31 / 12 / 4 tiles, 7.9 % band) | 163 ms | 131 ms | 172 ms |
| UHD coarse (49 / 18 / 9 tiles) | 261 ms | 207 ms | 401 ms |
| UHD all (85 / 29 / 12 tiles) | 451 ms | 328 ms | 529 ms |
| real t36, 2880×1620 (4.9 % band) | 248 ms | 168 ms | 262 ms |
| real t44, 2880×1620 (2.2 % band) | 133 ms | 92 ms | 91 ms |

Per tile 3.9 / 6.0 / 23.6 ms (small), 4.7 / 8.5 / 40.1 ms (base): **tile 512 is the best overall**; base is
about 1.2× small. Cold: model load 450-630 ms, first frame 290-730 ms. GPU memory after the run 1.4-3.2 GB
(small; base tile 1024 6.3 GB), RSS peak 1.2-1.7 GB. **BiRefNet**, whole frame: lite 112 ms HD / 251 ms UHD
(inference 57 ms; fp16 108 / 247), standard 158 / 297 (fp16 139 / 279); load 2.1-3.1 s; **GPU memory 8.3
(lite fp16) to 16.5 GB** (arena not limited), RSS ~2 GB; IoU 0.98 (lite) / 0.97 (standard); holes in the
model's mask 1-4 vs 7-11 in the truth (NOTE 2026-10-23, B3 evaluation: the truth's extra enclosed gaps were 1-px artefacts; on the P4 `gaps` scene BiRefNet keeps real gaps open (mean inside 0.01-0.12, 100/100 kept), so the earlier "closes real gaps" reading was wrong on synthetic data; real footage still unverified). **CPU** (Xeon): ViTMatte-small
tile 512 1.3 s HD, 2.7 s UHD, 2.2 s t36 / 1.2 s t44; BiRefNet-lite 2.9 s and 12 GB RSS. Frame times are about 10 %
above BFD's (a different GPU), the same shape. **Against decision 28's proposed gate:** with Flame idle the
worst case measured (UHD "all", tile 512) is 0.33 s and a typical HD frame 0.1-0.15 s, far inside 1 s / 3 s.
Still unmeasured: contention with Flame's own GPU work (the next test: a second instance while Flame renders)
and how the UI behaves when `render()` blocks for 0.1-0.4 s.

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

## Flame contention test on akflame5 (2026-10-23, user + lead)

`ofx/ml/package/contention.sh`: ViTMatte-small, CUDA, HD "all" scene, tile 512 (12 tiles), run in a
loop (30 warm frames per pass, a new session per pass) at 100 % duty while the user used Flame;
59 passes over ~7 minutes. Flame's own VRAM use is in the GPU column (our loop holds ~2.2 GB of it).
Reference without Flame: 131-134 ms.

| Phase | secs | passes | frame ms med (min-max) | GPU util avg/max | GPU MiB |
|---|---|---|---|---|---|
| Flame idle | 50 | 7 | 131.2 (131.0-131.7) | 28 / 45 % | 8.2-9.9 GB |
| playback (heavy setup, real time) | 245 | 35 | 134.0 (131.2-198.2) | 53 / 99 % | 8.5-44.8 GB |
| interactive (scrub, pan, params) | 47 | 7 | 132.5 (132.1-140.2) | 32 / 93 % | 22.0-23.6 GB |
| Batch Render | 55 | 7 | **189.8** (132.0-192.1) | 72 / 100 % | 22.0-24.2 GB |

User's observation: **no major slowdown in Flame** (no per-phase notes given). Reading: Flame idle,
playing back and scrubbing costs the model nothing measurable (a few passes reach 190-198 ms); a heavy
Batch Render on the same GPU slows it by ~45 % (131 → 190 ms). Flame's peak VRAM was ~45 GB (of 97.9
GB) including our 2.2 GB: lots of headroom. Per-pass session creation 455 ms median (max 904), first
frame 350-460 ms. **ONNX Runtime warns on every pass** that some nodes (the transformer's shape
handling) are not assigned to the CUDA EP and that 48 Memcpy nodes were added: normal for this export,
a candidate for optimisation (IoBinding, graph clean-up) but not a blocker. Not covered: the plugin
blocking Flame's own render thread (needs the phase 1 spike), a UHD pass, the user's exact setup.

## Mac (lbm, Apple Silicon, ORT 1.31 CoreML EP), ViTMatte, partial (2026-10-23; BiRefNet still running)

Warm frame median, `trimap` task, scene `coarse` (band 10 px at HD, 20 px at UHD). The ORT log says
CoreML takes only part of the graph (BiRefNet: 196 of 2034 nodes); the rest runs on the CPU.

| Model | Frame | Tile 256 | Tile 512 | Tile 1024 | Per tile run (256 / 512 / 1024) |
|---|---|---|---|---|---|
| small | HD coarse | 1.42 s | 2.39 s | 4.40 s | 82 / 258 / 1080 ms |
| small | HD all | 2.58 s | 3.17 s | 4.37 s | |
| small | UHD coarse | 4.10 s | 4.75 s | 9.80 s | |
| small | UHD all | 7.14 s | 7.57 s | 13.07 s | |
| base | HD coarse | 2.94 s | 4.92 s | 8.84 s | 172 / 541 / 2200 ms |
| base | UHD coarse | 8.60 s | 9.98 s | 20.03 s | |
| base | UHD all | 14.91 s | 15.81 s | 26.76 s | |

Load about 2 s, first frame = warm frame + 0.2-1.5 s. Reading: the CoreML path through ORT is 10-20x
slower than CUDA on akflame5 (HD coarse 90-170 ms) and misses the decision 28 gate (warm HD about 1 s)
by a wide margin even with the smallest tile; tile 256 is the best size here (per-tile cost grows
faster than tile area). Quality numbers are the same as on the CPU/CUDA runs (model-independent of EP).
Consequence for the Mac: either a native Core ML conversion (coremltools) is needed, or the Mac
plugin is a slow "it works" path. Open; to be decided after the full run.

## Colour probe in Flame 2027.2 (akflame5, 2026-10-23; OCIO style, aces2.0 config)

Same frame (t=5), LogC3 EI800 clip through a Resize, node Matte Refine ML, LTX disconnected:

| Request | Reported colourspace | RGB mean / max |
|---|---|---|
| none | ARRI LogC3 (EI800) | 0.460 / 0.843 |
| `Rec.1886 Rec.709 - Display` | Rec.1886 Rec.709 - Display | 0.682 / 3.11 |

- The request works through GetClipPreferences (`..._Source` out property: kOfxStatOK); on the clip
  descriptor in describe it is refused (kOfxStatErrUnknown).
- 0.843 -> about 2.95 and 0.298 -> about 0.31 by hand (LogC3 EI800 to linear, gamma 2.4): matches
  3.11 / 0.33 within the matrix difference: no tone mapping, no clamp. Values reach 3-4.
- With the LTX matte connected the Source clip's tag reads "Matte" (with or without the request)
  and the picture arrived already converted (0.682 / 3.11 in round 1 too): cause unexplained,
  the pixels were right. Raw clips tagged correctly: Rec.1886 Rec.709 - Display, ARRI LogC3 (EI800),
  Log3G10 REDWideGamutRGB.

### Result with the filter layout (decision 35; Flame 2027.2, akflame5, same frame, 2026-10-23)

| Clip | Picture Colour | Reported colourspace | RGB mean R / max R |
|---|---|---|---|
| Log3G10 | Bypass | Log3G10 REDWideGamutRGB | 0.277 / 0.835 (raw) |
| Log3G10 | Auto | Rec.1886 Rec.709 - Display | 0.430 / 4.41 (converted) |
| LogC3 | Bypass | ARRI LogC3 (EI800) | 0.460 / 0.843 (raw) |
| LogC3 | Auto | Rec.1886 Rec.709 - Display | 0.682 / 3.11 |
| Rec709 | either | Rec.1886 Rec.709 - Display | 0.572 / 1.0 (already display) |

With one mandatory Source clip and the matte in its alpha, Flame behaves as specified: Bypass raw with the
true tag, Auto converted with the requested name, the output colourspace answers are as coded, no "Matte"
tag on the picture. The two-clip layout (General context, Matte mask clip) was the cause of the earlier oddity.

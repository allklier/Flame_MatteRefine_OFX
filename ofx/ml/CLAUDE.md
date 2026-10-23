# ofx/ml/CLAUDE.md — Matte Refine ML (built; plan, decisions, measurements, risks)

The second plugin: ML and slow matte refinement, in this repo and CMake project next to Matte
Refine (root decisions 17, 18, 27). Plugin 1 stays real-time and inline (14). Goal, architecture and
the decision index are in the root `CLAUDE.md`; the operations plugin 2 reuses are in
`ofx/src/ops/CLAUDE.md`; Flame host rules in `ofx/CLAUDE.md`. Tags as there (DECIDED / PROPOSED /
OPEN / UNVERIFIED). Written 2026-10-23. Keep this file under ~250 lines.

## Scope (decision 17 holds)

The input is a good matte (LTX or elsewhere); every ML operation **keeps it and fixes defects**.
Whole-matting models (MatAnyone, VideoMaMa, GVM, SAM2Matting, SDMatte, keyers) stay out. The
user's picture (front input) is the guide, the matte (its alpha, decision 35) is the subject; same inputs
and colour rules as plugin 1 (decisions 22-24).

| # | Operation | What it does | Model (licences: `LICENCES.md`) |
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
  **Update 2026-10-23: forced by decision 32** (library collisions with Flame's own ONNX Runtime),
  but as a synchronous runner, not LTX's offline worker: decision 14 still holds.
  **Flame test 2026-10-23 (user): worked well, Flame stayed very responsive; the gate is met
  (see phase 1).** Inline-with-runner is the architecture; no Process button, no cache.
- **29. PROPOSED: runtime = ONNX Runtime** (in the runner process, 32; was: in the .ofx) (C API; CUDA execution provider on Rocky,
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
- **32. DECIDED (user, 2026-10-23): a synchronous runner process owns ORT, CUDA and the models;
  the .ofx holds no GPU library.** Why: Flame ships its own ONNX Runtime 1.22.0 and CUDA 12.8 /
  cuDNN 9.9 (`/opt/Autodesk/lib64/<version>/`, seen on akflame5) and can load them at any time
  (inference nodes, Isolate). Our ORT 1.31 / CUDA 13 in the same process would be shadowed by
  the soname `libonnxruntime.so.1`, and cuDNN's same-soname sub-libraries would mix with
  Flame's (order-dependent crashes in Flame's own nodes). Building against Flame's libraries
  (option B) was rejected: they change between Flame versions and we won't ship per-version
  plugins. Design: the plugin implements `TileRunner` by sending tiles to `mr_ml_runner` (one
  per user and machine, started lazily under a lock, shared by every node and every Flame
  process, one connection each, tiles in shared memory, one tile at a time on the GPU, idle
  exit); sessions live in the runner, one per model (LRU). Still synchronous per frame, no cache,
  no Process button. Protocol version handshake (a mismatched runner is told to quit and
  replaced); per-tile timeout; a runner failure gives a pass-through matte and an error footer,
  the next render restarts it. Mac: same design with CoreML. The spike's in-process path
  (`ort_runner`) stays as the runner's engine and the benchmark harness.
- **33. DECIDED (user, 2026-10-23): Matte Refine ML on macOS is deferred; the Mac ships Matte Refine
  (plugin 1) only.** Why: ViTMatte via ORT's CoreML EP is 10-20x slower than CUDA (HD 1.4-2.6 s,
  `PHASE0_RESULTS.md`), misses the decision 28 gate. The user leans towards a native Core ML conversion (coremltools) later, not now. The
  runner design (32) keeps the engine swappable, so a Mac runner can be added without changing the
  plugin. Rocky first.
- **34. DECIDED (user, 2026-10-23): "Picture Colour" = Auto / Bypass; the plugin does no colour maths.**
  Auto asks Flame for `Rec.1886 Rec.709 - Display` (OpenFX 1.5 `OfxImageClipPropPreferredColourspaces_Source`
  in the GetClipPreferences out args; built-in list `kPicturePref`);
  Bypass asks for nothing. Model input = picture clamped to 0-1, no gamma (synthetic band error
  better on every scene). Verified in Flame 2027.2 (`PHASE0_RESULTS.md`, "Colour probe"): Flame
  converts (LogC3 mean 0.46 -> 0.68, plain gamma 2.4, no tone mapping, values above 1), the
  descriptor route is refused, the reported tag reads "Matte" when a matte is connected (never
  decide from it). Live switching Auto/Bypass VERIFIED in Flame (clip preferences re-run on the flip). In the
  old two-clip layout the picture arrived converted in both modes (35); in the filter layout both modes
  behave as specified. Fallback if it ever fails: OCIO in the runner (explicit dropdown).
- **35. DECIDED (user, 2026-10-23): Matte Refine ML is a filter, like LTX Alpha Gen: picture required on
  the front, the matte in its alpha. Supersedes decision 22 for ML only (plugin 1 keeps it).** Why: with
  the separate Matte mask clip (General context, Source optional) Flame 2027.2 delivered the picture
  converted to Rec.1886 even in Bypass and tagged the Source "Matte" (`PHASE0_RESULTS.md`, Colour probe;
  cause unexplained); LTX, a plain filter, does not. Decision 22's reason (a node without a picture)
  does not apply: the model needs the picture. Built: `eContextFilter` only, mandatory Source
  (RGBA/RGB), no Matte clip, RoD from Source; footers for a constant alpha ("connect the matte to the
  node's second socket") and an RGB-only source. Output tag: Auto = the requested name, Bypass =
  Source. The temporary developer keys (`ml_layout`, `ml_output_cs`, `ml_picture_pref`) and the legacy
  layout were removed 2026-10-23. **VERIFIED in Flame 2027.2 (user + log, 2026-10-23): the conversion / "Matte" tag problem is gone**
  (Bypass = raw clip with its true tag, Auto = converted with the requested name; user: input layout matches
  LTX). The separate Matte mask clip caused it. UNVERIFIED: old setups across layouts, Flame's describe cache.
- **36. DECIDED (user, 2026-10-23): feature plan** (`PLAN_FEATURES.md`). B2 DiffMatte is a model choice
  inside Edge Refine, not its own column; a slow model is acceptable behind a Quality switch if it passes
  the quality test; BiRefNet must run in 24 GB of VRAM: limit its arena only when the card is small
  (about 32 GB or less), never cap bigger cards; build order B3 (smart fill) before B4 (temporal).
  **Update 2026-10-23: B2 DiffMatte SHELVED (user): no licence for code or weights, weights only on Google Drive;
  rule: fetch weights only from stable hosts such as HuggingFace. Spike kept in `research/diffmatte/`.**

## Plan, in order (each phase ends with a go / no-go for the next)

0. **Benchmark first, no plugin. DONE 2026-10-23 on BFD and akflame5 incl. the Flame contention test (results: summary below, details in `PHASE0_RESULTS.md`); the Mac run is in progress.** A standalone tool `ofx/ml/tools/mr_ml_bench`: loads an ONNX
   file with the candidate runtime and runs tiles cut from the synthetic scenes (21) and the
   user's real mattes (the Save Matte dumps; confidential, stay on the machine) at HD and UHD.
   Measure: warm and cold latency per tile (256, 512, 1024), tiles per typical frame, total
   per-frame time, VRAM, model load time, host RAM; on akflame5 (RTX PRO 6000 Blackwell) and lbm
   (M1 Ultra); and **while Flame is rendering** (GPU contention, the LTX nodes' "unload GPU"
   lesson). Candidates: ViTMatte small and base (fp32, fp16), BiRefNet lite and standard.
   Output: a table here, and the quality of each on the synthetic "coarse" / "eroded" scenes
   against their truth (B1) and on a real frame.
1. **Spike plugin with the runner (decision 32). BUILT 2026-10-23 (Opus agents), tests green on BFD
   and akflame5 (21/21); not yet tried in Flame.** Node `MatteRefineML` 0.0.1 (spike; now 0.1.0), B1 only, ViTMatte
   small / base; columns Matte (View Picture/Matte/Trimap/Changes) · Edge Refine (Enable, Band px
   10, Mix, Only Grey) · Model & Support. `render()` stays synchronous (14). The .ofx holds no ORT
   and loads no GPU library (test `ml_bundle_deps`: about 770 KB, libc/libstdc++ only, no RPATH).
   Code in `ofx/ml/src/`: `edge_refine` (core, `TileRunner`), `runner_client` (in the plugin),
   `runner_ipc` / `runner_protocol` / `runner_server` / `runner_main` (`mr_ml_runner`, C++17,
   owns ORT via `ort_runner`).
   - **Runner:** one per user and machine, shared by all nodes and Flame processes; socket
     `$XDG_RUNTIME_DIR/matte-refine-ml/runner.sock` (else `/tmp/matte-refine-ml-<uid>/`, mode
     0700, peer uid checked); started lazily by the first tile under `flock`, via `posix_spawn`
     with a fixed environment (no Flame `LD_LIBRARY_PATH` / `LD_PRELOAD`), daemonises, re-execs
     itself with `LD_LIBRARY_PATH` = ORT folder + `ml_cuda_libs`. Protocol v1: 16-byte headers,
     HELLO passes a sealed memfd slot (tiles in shared memory), RUN / STATUS / QUIT; build id =
     version + git hash, a different build is told to QUIT and replaced once. Sessions per
     (model, EP), shared, max 2 (LRU), failures remembered 30 s; model calls FIFO, one at a time;
     idle exit 5 min; `mr_ml_runner --status|--quit`. **After rebuilding with the same git hash
     run `--quit`.**
   - **Runner protocol 2 + model table (P1, 2026-10-23).** `model_table.{h,cpp}`: a row per model (id, file
     `<id>/model.onnx`, tensor names, channels in / out, size rule, licence id, sha256 (unchecked until P2),
     CUDA `gpuMemLimitMiB` / `arenaSameAsRequested`; rows: vitmatte-small / base, birefnet-lite / birefnet fp32).
     RUN names the model and carries {cIn, H, W, cOut, Ho, Wo, x, y}; slot = [input][output] float32 (20 MiB,
     grows by reconnect, cap 256 MiB); the server validates every field (kBadRequest + reason). Client:
     generic `run(model, ...)`, `TileRunner` on top (ViTMatte bit-identical on CPU), `runFrame` for B3. BFD:
     no slowdown (HD 118 ms); BiRefNet-lite whole frame IoU 0.983, 155 ms; arena limit works (lite: unlimited
     15.7 GB, 12 GB limit -> 12.6, 8 GB -> 8.2 and runs, 4 GB -> clean error): data for decision 36.
   - **B3 Smart Fill built into the plugin (2026-10-23; not yet tried in Flame).** Column Smart Fill between Matte and
     Edge Refine (Enable off, Subject Threshold 0.6, Max Hole % 10, Segmentation Lite / Full), chain matte -> Smart Fill ->
     Edge Refine (`render_chain.{h,cpp}`; a failing step passes its input through with a footer), View "Holes" (filled green,
     kept red). Decision 36 memory rule in the runner (`cudaArenaFor`): arena hint only on cards <= 32768 MiB total; hints
     lite 8192, full 10240 (BFD: full fails at 8192, lite at 6144). BFD warm: HD 148-160 ms (model 121), UHD 209 ms;
     lite + Edge Refine HD 299 ms. Smart Fill off = previous build byte for byte. Details `PLAN_FEATURES.md`.
   - **B4 Temporal built into the plugin (2026-10-23; not yet tried in Flame).** Column Temporal before Smart Fill (Enable off, Frames
     +-1 / +-2, Strength 1, Band px 6 x render scale); chain matte -> Temporal -> Smart Fill -> Edge Refine; temporal clip access declared on
     the effect and Source; `getFramesNeeded` asks [t-K, t+K]; neighbours via `fetchImage(t+-k)` in render(); a failure or no neighbours passes
     the matte through with a footer. Scratch host: HD K=1 flicker -56 %, K=2 -69 %, nothing lowered outside the band, Temporal off byte-identical.
     Plugin cost about 75 / 130 ms (HD K=1 / 2), 210 / 365 ms (UHD) plus Flame's fetch per neighbour. UNVERIFIED in Flame: fetchImage(t+-k) in
     render(), getFramesNeeded honoured, fetch cost, cache invalidation (the log answers: `TEMPORAL_TEST.txt`).
   - **Failures:** a broken connection is retried once; a tile unanswered within
     `ml_tile_timeout_s` (20) is an error; the footer says why, the matte passes through, the next
     render retries. Install: runner in `<bundle>/Contents/Resources/` (`ml_runner` overrides).
     Config: plugin `ml_runner`, `ml_tile_timeout_s`; runner `ml_model_dir`, `ml_ep`,
     `ml_cuda_libs`, `log_level`.
   - **Measured on BFD:** warm HD 124 ms, UHD 298 ms (same as in-process); transport about 0.3 ms
     per tile; two clients 206-215 ms per HD frame each; cold first frame 1.25-1.4 s; runner RSS
     1.1 GB, GPU +1.5 GB; CPU EP bit-identical to in-process, CUDA within 1.7e-3. Verified with
     a dummy `libonnxruntime.so.1` preloaded into the host and `kill -9` of the runner mid-frame.
   - **Flame test on akflame5, 2026-10-23 (user + log): worked well, Flame stayed very responsive.**
     Log of 152 active renders (HD, two nodes, one runner, one start, no reconnect, no abort, no
     error, no frame rendered twice): render() median 105 ms (Small) / 111 ms (Base), p95 108 /
     175 ms; cold first frame 789 ms (Small), 587 ms (Base, second session 437 ms); runner
     overhead about 2 ms; 978 tiles served. Not in the log: UHD, a long Batch Render, a killed runner.
   - **B1 polish (2026-10-23, built, not yet in Flame).** Edge Refine = Enable, Band px 10, **Outer px (0)**, Mix, Only Grey (5 rows). Outer px is
     an asymmetric reach on the background side (unknown region -(band + outer) <= sd <= band; both ends fade over band/2; 0 = the old rule bit for
     bit). Hair at 1080: Outer 100-150 recovers about 5x the lost strand alpha (0.13 -> 0.67) for about +25 % frame time, but costs quality on clean
     mattes (fgcolor, eroded, mixed_edge) and spills on busy backgrounds from 190: default stays 0, keep it <= about 150 frame px at any resolution;
     a wider symmetric band does nothing for hair; picture-guided reach rejected. Band / Outer px stay in frame pixels (x render scale). Runner:
     a hung runner is probed (STATUS, 1 s) and killed only when the peer pid and uid match; socket `runner-<8 hex of build id>.sock` per build;
     runner logs as component "runner" (`mr::setLogComponent`, ofx/src/log); `runner-stderr.log` capped at 5 MiB. **OPEN:** the threaded trimap
     (`signedEdgeDistanceThreaded`: HD -8 ms, UHD -16..-50 ms, but the band border moves up to 1 px and per-pixel results exceed the model noise: not
     switched; the user decides); GPU 0 only; Collect Logs may not include `runner-stderr.log` (unchecked).
2. **Decision 28's gate:** stay inline / add a worker with Process + cache / both (inline for
   the light models, worker for DiffMatte). **PROPOSED thresholds** (the user sets them): a warm
   HD frame of a typical subject within about 1 s and no single frame longer than about 3 s
   inline; otherwise a worker. UNVERIFIED: whether Flame caches OFX results between frames
   (Neat-style "middle ground"): test it in phase 1.
3. B3 (BiRefNet guided fill), then B2 (DiffMatte) on the chosen architecture.
4. B4 (temporal): needs neighbour frames (`setTemporalClipAccess`, UNVERIFIED in Flame) and
   probably the cache; last.
5. Packaging: install script, model setup and licence gate, release notes, Mac build.

## Phase 0 results: summary (details: `PHASE0_RESULTS.md`)

Tool `ofx/ml/tools/mr_ml_bench` (Opus agent, 2026-10-23), run package `ofx/ml/package/`. Measured on BFD
(RTX PRO 4500), **akflame5 (RTX PRO 6000 Blackwell, Flame's driver 580.159, Xeon 658X)** and, in progress,
lbm (M1 Ultra). Everything below is on akflame5 unless noted.
- **Speed (ViTMatte-small, CUDA, tile 512, band 10 px, all edge tiles of a frame in sequence):** HD frame
  98-131 ms, UHD 207-328 ms, real t36 (2880×1620) 168 ms, t44 92 ms; base model 1.2× slower; per tile 6 ms
  (512), 24 ms (1024: 4× the cost per pixel). Cold: load ~0.5 s, first frame 0.3-0.7 s. GPU memory 1.4-3.2
  GB. CPU only: 1.3 s HD / 2.7 s UHD (a fallback, not a plan). BiRefNet whole frame 112 ms HD / 251 ms UHD,
  8-16 GB of GPU memory (arena unlimited), IoU 0.97-0.98, (B3 eval 2026-10-23: keeps real gaps open on synthetic scenes; the Phase 0 "closes real gaps" note counted 1-px truth artefacts; real footage unverified).
- **Quality (synthetic, mean error in the band; input / ViTMatte / Snap to Image):** coarse 0.221 / 0.033 /
  0.095, eroded 0.118 / 0.010 / 0.076, all 0.079 / 0.020-0.026 / 0.063; it makes a clean matte worse (0.013),
  removes hair outside the 10 px band, and the synthetic picture contains the true edge: real footage decides.
- **Runtime:** ONNX Runtime 1.31 + CUDA 13.4 libraries (PyPI wheels, 1.6 GB) **run under Flame's driver
  580** (verified); needs glibc 2.28 only (fits Rocky 9); the headers need **C++17** (own translation
  units; the plugin is C++14); macOS package includes the CoreML EP.
- **Contention with Flame (user test):** no major slowdown in Flame; the model is unaffected by idle,
  playback and scrubbing and ~45 % slower (131 → 190 ms) during a heavy Batch Render. Flame peaks ~45 GB
  VRAM: headroom is not a problem.
- **Against the proposed gate (decision 28):** met by a wide margin for GPU-sharing and time. **Not yet
  measured: the plugin blocking Flame's render thread** (phase 1 spike) and the Mac.
## Licences

Moved to `LICENCES.md` (table, precedent, Baselight findings). Decision 31 stays here.

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

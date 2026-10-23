mr_ml_bench run package (Matte Refine ML, phase 0; ofx/ml/CLAUDE.md)

The tool's source is in the repo copy: ofx/ml/tools/mr_ml_bench.cpp, ofx/ml/CMakeLists.txt, and
the guarded MR_BUILD_ML option in the root CMakeLists.txt. It links mr_ops and mr_synth_lib from
the same repo. Scripts here: fetch.sh, build_rocky9.sh, build_mac.sh, run_bench.sh.

Downloads (fetch.sh checks every sha256; nothing comes from the developer's machine):
  ONNX Runtime 1.31.0  Linux: onnxruntime-linux-x64-gpu_cuda13-1.31.0.tgz  (245 MB, 305 MB unpacked)
                       macOS: onnxruntime-osx-arm64-1.31.0.tgz (37 MB; CoreML EP included)
  CUDA 13 libs (Linux, PyPI wheels, unpacked to cuda_libs/, 1.6 GB): nvidia-cublas 13.8.1.7,
    nvidia-cuda-runtime 13.4.92, nvidia-curand 10.4.4.72, nvidia-cudnn-cu13 9.27.0.42
  Models (1.9 GB): vitmatte-small (104 MB), vitmatte-base (387 MB), birefnet-lite fp32 (224 MB)
    + fp16 (115 MB), birefnet fp32 (973 MB) + fp16 (490 MB). URLs and sha256 in fetch.sh.
  Licences: ONNX Runtime MIT; BiRefNet MIT; ViTMatte card says apache-2.0 but is "trained on
    Composition-1k" (Adobe data, non-commercial terms): evaluation only until decided.

akflame5 (Rocky 9, GCC 11): rsync the repo copy (e.g. to ~/MatteRefine_build/mlbench; check
/home's free space first: ~3.5 GB for the downloads, or put them on another disk), then
  D=<download dir>; package/fetch.sh $D
  package/build_rocky9.sh <repo copy> $D/ort/onnxruntime-linux-x64-gpu_cuda13-1.31.0
  BENCH=<repo copy>/build_ml/ofx/ml/mr_ml_bench CUDA_LIBS=$D/cuda_libs \
    DUMPS="$HOME/.cache/matte-refine/dumps/05a4e946_t36.f32:2880x1620 $HOME/.cache/matte-refine/dumps/05a4e946_t44.f32:2880x1620" \
    package/run_bench.sh $D/models $D/ort/onnxruntime-linux-x64-gpu_cuda13-1.31.0 ~/mlbench_out
  Contention test: start a second run_bench.sh (or one mr_ml_bench with --repeat 200) while
  Flame renders, and compare bench.jsonl.

lbm (macOS arm64): same with fetch.sh (no CUDA libs), build_mac.sh <repo copy> <ort dir>
  [<LTX's cmake>], run_bench.sh <models> <ort dir> <out> "coreml cpu". The CoreML EP and the Mac
  build are UNTESTED (written on Linux). nvidia-smi is absent: GPU MiB reads -1; RSS reads -1
  (no /proc): use Activity Monitor or `footprint` if memory matters.

Output: <out>/bench.jsonl (one JSON object per run), <out>/summary.txt (readable lines + quality),
<out>/stderr.txt (ORT warnings). Real mattes: timing only, the picture is synthetic noise; only
aggregate numbers are printed. Runtime here: 2-8 s per run (23 runs in 138 s); the full
matrix is ~130 runs, about 10-15 min (QUICK=1: coarse + tile 512 only). Each line = one process: session
creation (load), first frame (cold), --repeat warm frames.

Tool usage (see the header of mr_ml_bench.cpp):
  mr_ml_bench --model m.onnx --task trimap|segment --ep cuda|cpu|coreml --tile 256|512|1024
    --scene coarse|eroded|mixed_edge|fgcolor|all|... --size WxH | --matte f.f32 --size WxH
    [--band 10 (px at 1080 lines)] [--overlap 32] [--repeat 5] [--json out.jsonl] [--dump dir]
  mr_ml_bench --model m.onnx --info      (input / output names, shapes, types)
  mr_ml_bench --selfcheck                (every unknown pixel is covered by a selected tile)

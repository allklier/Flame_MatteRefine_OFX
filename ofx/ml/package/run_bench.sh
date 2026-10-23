#!/bin/bash
# run_bench.sh <models_dir> <ort_dir> [out_dir] [ep list]   (mr_ml_bench matrix, < 10 min on a GPU)
# Env: BENCH=<mr_ml_bench binary> (default ../../build/ofx/ml/mr_ml_bench next to this package's repo),
#      CUDA_LIBS=<dir with libcudart.so.13, libcublas*, libcurand, libcudnn*> (Linux CUDA EP),
#      DUMPS="file1.f32:WxH file2.f32:WxH" (real mattes, timing only), QUICK=1 (fewer runs).
set -u
MODELS=$(cd "$1" && pwd); ORT=$(cd "$2" && pwd); OUT=${3:-$PWD/bench_out}; EPS=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$HERE/../../build/ofx/ml/mr_ml_bench}
mkdir -p "$OUT"; JSON=$OUT/bench.jsonl; LOG=$OUT/summary.txt; : > "$JSON"; : > "$LOG"
case "$(uname)" in
  Darwin) export DYLD_LIBRARY_PATH=$ORT/lib; EPS=${EPS:-"coreml cpu"} ;;
  *) export LD_LIBRARY_PATH=$ORT/lib${CUDA_LIBS:+:$CUDA_LIBS}; EPS=${EPS:-"cuda cpu"} ;;
esac
{ echo "host $(hostname) $(uname -sm) $(date)"; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  grep -m1 'model name' /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null; } >> "$LOG"
run() { echo "+ $*" >> "$LOG"; "$BENCH" --json "$JSON" "$@" 2>>"$OUT/stderr.txt" | grep -v '^$' >> "$LOG"; }
SIZES="1920x1080 3840x2160"
for EP in $EPS; do
  if [ "$EP" = cpu ]; then REP=1; TILES="512"; SCENES="coarse"; else REP=5; TILES="256 512 1024"; SCENES="coarse eroded mixed_edge fgcolor all"; fi
  [ -n "${QUICK:-}" ] && { TILES=512; SCENES=coarse; }
  for M in vitmatte-small/model.onnx vitmatte-base/model.onnx; do
    [ -f "$MODELS/$M" ] || { echo "missing $M" >> "$LOG"; continue; }
    [ "$EP" = cpu ] && [ "$M" = vitmatte-base/model.onnx ] && continue
    for S in $SIZES; do for T in $TILES; do for SC in $SCENES; do
      run --model "$MODELS/$M" --task trimap --ep $EP --scene $SC --size $S --tile $T --repeat $REP
    done; done; done
    for D in ${DUMPS:-}; do for T in $TILES; do
      run --model "$MODELS/$M" --task trimap --ep $EP --matte "${D%%:*}" --size "${D##*:}" --tile $T --repeat $REP
    done; done
  done
  for M in birefnet-lite/model.onnx birefnet-lite/model_fp16.onnx birefnet/model.onnx birefnet/model_fp16.onnx; do
    [ -f "$MODELS/$M" ] || { echo "missing $M" >> "$LOG"; continue; }
    [ "$EP" = cpu ] && [ "$M" != birefnet-lite/model.onnx ] && continue
    for S in $SIZES; do for SC in all coarse; do
      [ "$EP" = cpu ] && [ $SC = coarse ] && continue
      run --model "$MODELS/$M" --task segment --ep $EP --scene $SC --size $S --repeat $REP
    done; done
  done
done
echo "done: $JSON, $LOG"

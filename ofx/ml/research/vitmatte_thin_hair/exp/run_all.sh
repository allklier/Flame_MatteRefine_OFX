#!/bin/bash
# Runs the synthetic thin-strand study: per scene generate jobs, run ViTMatte, evaluate, delete inputs.
# Usage: PY=<python with numpy+scipy> ORT=<ort dir> CUDA_LIBS=<dir> run_all.sh <model.onnx> <outdir> [first] [last] [cpu]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=$1; OUT=$2; FIRST=${3:-0}; LAST=${4:-$(( $("$PY" "$HERE/synth.py" list) - 1 ))}; EP=${5:-}
export LD_LIBRARY_PATH=$ORT/lib${CUDA_LIBS:+:$CUDA_LIBS}
mkdir -p "$OUT"
for i in $(seq "$FIRST" "$LAST"); do
  J=$OUT/scene_$i
  "$PY" "$HERE/synth.py" gen "$i" "$J"
  "$HERE/vm_run" "$MODEL" "$J" $EP
  "$PY" "$HERE/synth.py" eval "$J" "$OUT/results.csv"
  rm -f "$J"/job_*.in "$J"/job_*.out
done

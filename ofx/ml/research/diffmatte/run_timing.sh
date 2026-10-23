#!/bin/bash
# Timing through mr_ml_bench, CUDA EP, tile 512, band 10 px, warm frames (repeat 5). Run with the
# GPU otherwise idle; logs the other GPU users before each run.  run_timing.sh <B> <vitmatte dir>
set -u
B=$1; VM=$2; BENCH=$B/build/ofx/ml/mr_ml_bench; OUT=$B/out/timing; mkdir -p $OUT
export LD_LIBRARY_PATH=${CUDA_LIBS:?set CUDA_LIBS}
models="vitmatte-small=$VM/vitmatte-small/model.onnx vitmatte-base=$VM/vitmatte-base/model.onnx"
for m in vits-s1 vits-s3 vits-s5 vits-s10 vitb-s1 vitb-s5 vitb-s10; do models="$models dm-$m=$B/models/diffmatte-$m/model.onnx"; done
for mm in $models; do name=${mm%%=*}; file=${mm#*=}
  for ss in coarse:1920x1080 all:1920x1080 hair_busy:1920x1080 all:3840x2160; do sc=${ss%%:*}; sz=${ss#*:}
    others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
    line=$($BENCH --model $file --task trimap --ep cuda --scene $sc --size $sz --band 10 --tile 512 --repeat 5 \
      --json $OUT/timing.jsonl 2>&1 | grep 'warm frame' | head -1)
    echo "$name $sc $sz | others: $others | $line"
  done; done

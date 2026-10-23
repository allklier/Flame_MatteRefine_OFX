#!/bin/bash
# Quality matrix through mr_ml_bench (CUDA EP), dumps per run for score.py.
#   run_quality.sh <B=ml_work_b2> <vitmatte model dir>
set -u
B=$1; VM=$2; BENCH=$B/build/ofx/ml/mr_ml_bench; OUT=$B/out/quality; mkdir -p $OUT
export LD_LIBRARY_PATH=${CUDA_LIBS:?set CUDA_LIBS}
models="vitmatte-small=$VM/vitmatte-small/model.onnx vitmatte-base=$VM/vitmatte-base/model.onnx"
for m in vits-s1 vits-s3 vits-s5 vits-s10 vits-s10-seed1 vitb-s1 vitb-s5 vitb-s10 vits1024-s10 vits1024-s1; do
  models="$models dm-$m=$B/models/diffmatte-$m/model.onnx"; done
for mm in $models; do name=${mm%%=*}; file=${mm#*=}
  for sb in hair_flat:10 hair_busy:10 hair_flat:30 hair_busy:30 coarse:10 eroded:10 mixed_edge:10; do
    sc=${sb%%:*}; band=${sb#*:}; d=$OUT/$name/${sc}_b$band; mkdir -p $d
    [ -f $d/done ] && continue
    $BENCH --model $file --task trimap --ep ${EP:-cuda} --scene $sc --size 1920x1080 --band $band --tile 512 \
      --repeat 1 --dump $d --json $OUT/quality.jsonl > $d/log.txt 2>&1 && touch $d/done
    echo "$name $sc b$band: $(grep -o 'ml_mean = [0-9.e-]*' $d/log.txt)"
  done; done

#!/bin/bash
# contention.sh: runs the ViTMatte benchmark in a loop on the GPU while you use Flame, and logs GPU
# use. Commands (all quick; the loop runs in the background):
#   contention.sh start [hd|uhd]   start the loop (hd = 1920x1080 "all" scene, uhd = 3840x2160) + GPU log
#   contention.sh mark "text"      write a labelled timestamp (use it at the start/end of each phase)
#   contention.sh status           what is running, the last results
#   contention.sh stop             stop the loop and the GPU log
# Output: $B/contention/{timeline.log,gpu.csv,marks.log,stderr.txt}. One loop iteration = one process:
# session creation, then 30 frames (ViTMatte-small, CUDA, tile 512, all tiles of the frame one after
# another, like an inline plugin would), median reported. See ofx/ml/CLAUDE.md "Phase 0".
B=${MLBENCH:-$HOME/MatteRefine_build/mlbench}
D=$B/data
ORT=$D/ort/onnxruntime-linux-x64-gpu_cuda13-1.31.0
BENCH=$B/repo/build_ml/ofx/ml/mr_ml_bench
OUT=$B/contention
mkdir -p "$OUT"
alive() { [ -f "$OUT/$1.pid" ] && kill -0 "$(cat "$OUT/$1.pid")" 2>/dev/null; }

case "${1:-}" in
  start)
    alive loop && { echo "already running (contention.sh stop first)"; exit 1; }
    size=1920x1080; [ "${2:-hd}" = uhd ] && size=3840x2160
    : > "$OUT/timeline.log"; : > "$OUT/marks.log"; : > "$OUT/stderr.txt"; rm -f "$OUT/bench.jsonl"
    touch "$OUT/run"
    nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 \
      > "$OUT/gpu.csv" 2>/dev/null < /dev/null &
    echo $! > "$OUT/smi.pid"
    nohup bash -c "
      export LD_LIBRARY_PATH=$ORT/lib:$D/cuda_libs
      while [ -f '$OUT/run' ]; do
        '$BENCH' --model '$D/models/vitmatte-small/model.onnx' --task trimap --ep cuda --scene all \
          --size $size --tile 512 --repeat 30 --json '$OUT/bench.jsonl' > /dev/null 2>> '$OUT/stderr.txt'
        echo \"\$(date +%s.%N) \$(tail -n 1 '$OUT/bench.jsonl' | cut -c1-600)\" >> '$OUT/timeline.log'
      done" > /dev/null 2>&1 < /dev/null &
    echo $! > "$OUT/loop.pid"
    echo "started ($size). Use: contention.sh mark \"what you are doing\"   and   contention.sh stop" ;;
  mark)
    echo "$(date +%s.%N) ${2:-mark}" >> "$OUT/marks.log"; echo "marked $(date +%T): ${2:-mark}" ;;
  status)
    alive loop && echo "loop running" || echo "loop not running"
    echo "iterations: $(wc -l < "$OUT/timeline.log" 2>/dev/null)   marks: $(wc -l < "$OUT/marks.log" 2>/dev/null)"
    tail -n 1 "$OUT/gpu.csv" 2>/dev/null
    tail -n 1 "$OUT/timeline.log" 2>/dev/null | grep -o '"frame_ms_median":[0-9.]*' ;;
  stop)
    rm -f "$OUT/run"
    for i in $(seq 1 60); do alive loop || break; sleep 0.5; done
    alive loop && kill "$(cat "$OUT/loop.pid")" 2>/dev/null
    alive smi && kill "$(cat "$OUT/smi.pid")" 2>/dev/null
    echo "stopped. iterations: $(wc -l < "$OUT/timeline.log")  marks: $(wc -l < "$OUT/marks.log")  (results in $OUT)" ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac

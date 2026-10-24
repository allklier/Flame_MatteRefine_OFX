#!/bin/sh
# Decision 32: MatteRefineML.ofx links no ONNX Runtime and no CUDA library (they live in
# mr_ml_runner), exports only the two OFX entry points, and stays small. Linux (readelf, nm).
# Usage: check_bundle_deps.sh <MatteRefineML.ofx>
set -u
ofx="$1"
fail=0
[ -f "$ofx" ] || { echo "no such file: $ofx"; exit 1; }
needed=$(readelf -d "$ofx" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
echo "NEEDED: $(echo $needed)"
for lib in $needed; do
  case "$lib" in
    libc.so.*|libm.so.*|libstdc++.so.*|libgcc_s.so.*|libdl.so.*|libpthread.so.*|ld-linux*) ;;
    *) echo "FAIL: unexpected dependency $lib"; fail=1 ;;
  esac
done
rpath=$(readelf -d "$ofx" | grep -E '\((RPATH|RUNPATH)\)')
[ -z "$rpath" ] || { echo "FAIL: the plugin has a library path: $rpath"; fail=1; }
# It cannot load a library at run time: no dlopen / dlmopen / dlsym import (dladdr is fine). The CUDA
# library names may appear: the setup gate (gate.h) checks that those files exist, nothing loads them.
dl=$(nm -D --undefined-only "$ofx" | awk '{print $2}' | sed 's/@.*//' | grep -E '^(dlopen|dlmopen|dlsym|dlvsym)$')
[ -z "$dl" ] || { echo "FAIL: the plugin imports $(echo $dl): it could load ORT / CUDA"; fail=1; }
if strings -a "$ofx" | grep -E 'libonnxruntime|OrtGetApiBase' >/dev/null; then
  echo "FAIL: the plugin names an ORT library or symbol:"
  strings -a "$ofx" | grep -E 'libonnxruntime|OrtGetApiBase' | head -5
  fail=1
fi
exports=$(nm -D --defined-only "$ofx" | awk '$2 ~ /^[TDBVWRiu]$/ {print $3}' | sed 's/@.*//' | sort | tr '\n' ' ')
echo "exports: $exports"
[ "$exports" = "OfxGetNumberOfPlugins OfxGetPlugin " ] || { echo "FAIL: exports differ from the two OFX entry points"; fail=1; }
size=$(wc -c < "$ofx")
echo "size: $size bytes"
[ "$size" -lt 3000000 ] || { echo "FAIL: larger than 3 MB"; fail=1; }
[ $fail = 0 ] && echo "check_bundle_deps: ok"
exit $fail

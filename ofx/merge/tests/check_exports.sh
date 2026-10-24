#!/bin/sh
# MatteMerge.ofx exports only the two OFX entry points and needs only the C / C++ runtime (as
# Matte Refine: libstdc++ dynamic, no other library). Linux (readelf, nm).
# Usage: check_exports.sh <MatteMerge.ofx>
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
exports=$(nm -D --defined-only "$ofx" | awk '$2 ~ /^[TDBVWRiu]$/ {print $3}' | sed 's/@.*//' | sort | tr '\n' ' ')
echo "exports: $exports"
[ "$exports" = "OfxGetNumberOfPlugins OfxGetPlugin " ] || { echo "FAIL: exports differ from the two OFX entry points"; fail=1; }
[ $fail -eq 0 ] && echo "check_exports: OK"
exit $fail

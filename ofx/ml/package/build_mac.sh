#!/bin/bash
# build_mac.sh <repo copy> <ort_dir> [cmake]: as build_rocky9.sh with clang and the Command Line
# Tools SDK (macOS 26: the same SDK flags as scripts/build_mac.sh). UNTESTED (no Mac here).
set -eu
REPO=$(cd "$1" && pwd); ORT=$(cd "$2" && pwd); CMAKE=${3:-cmake}
SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
"$CMAKE" -S "$REPO" -B "$REPO/build_ml" -DCMAKE_BUILD_TYPE=Release -DMR_BUILD_ML=ON -DORT_DIR="$ORT" \
  -DCMAKE_OSX_SYSROOT=$SDK -DCMAKE_CXX_FLAGS="-stdlib++-isystem $SDK/usr/include/c++/v1"
"$CMAKE" --build "$REPO/build_ml" -j8 --target mr_ml_bench
"$REPO/build_ml/ofx/ml/mr_ml_bench" --selfcheck
echo "built: $REPO/build_ml/ofx/ml/mr_ml_bench"

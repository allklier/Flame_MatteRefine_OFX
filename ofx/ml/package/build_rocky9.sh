#!/bin/bash
# build_rocky9.sh <repo copy> <ort_dir>: builds mr_ml_bench with the system GCC 11 (Rocky 9) into
# <repo copy>/build_ml and runs its ctest (tiling self-check; the smoke test needs MR_ML_MODELS).
set -eu
REPO=$(cd "$1" && pwd); ORT=$(cd "$2" && pwd)
cmake -S "$REPO" -B "$REPO/build_ml" -DCMAKE_BUILD_TYPE=Release -DMR_BUILD_ML=ON -DORT_DIR="$ORT" ${MR_ML_MODELS:+-DMR_ML_MODELS=$MR_ML_MODELS}
cmake --build "$REPO/build_ml" -j8 --target mr_ml_bench
"$REPO/build_ml/ofx/ml/mr_ml_bench" --selfcheck
echo "built: $REPO/build_ml/ofx/ml/mr_ml_bench"

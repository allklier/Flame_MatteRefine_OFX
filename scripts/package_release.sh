#!/usr/bin/env bash
# Release packages for Linux, built on the Rocky 9 build machine (glibc 2.34, GCC 11: what Flame
# loads), in the source copy scripts/build_remote.sh syncs there:
#   ORT_DIR=<extracted onnxruntime-linux-x64-gpu_cuda13-1.31.0> scripts/package_release.sh <build dir>
# Configures and builds <build dir> (Release, MR_BUILD_ML=ON, MRML_BUNDLE_ORT=ON: ONNX Runtime inside
# the ML bundle, the runner's RUNPATH $ORIGIN/lib only; MR_BUILD_MERGE=ON), runs ctest, stages and
# writes into dist/:
#   MatteRefine-<MR_VERSION>-linux-x86_64.tar.gz      Matte Refine (plugin 1), install.sh, docs
#   MatteRefineML-<MRML_VERSION>-linux-x86_64.tar.gz  Matte Refine ML with ONNX Runtime, install.sh,
#                                                     setup.sh, licences.txt, docs (no models, no CUDA)
#   MatteMerge-<MM_VERSION>-linux-x86_64.tar.gz       Matte Merge (plugin 3), install.sh, docs
#   SHA256SUMS
# MR_GIT=<hash> names the build (the copy on the build machine has no .git; build_remote.sh's way).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:?usage: ORT_DIR=<onnxruntime dir> scripts/package_release.sh <build dir>}"
ORT_DIR="${ORT_DIR:?set ORT_DIR to the extracted ONNX Runtime 1.31.0 CUDA 13 package}"
ORT_DIR="$(cd "$ORT_DIR" && pwd)"
DIST="${DIST:-$SRC/dist}"  # DIST=<folder> elsewhere (tests)

# The versions: the set() lines of the CMake files.
MR_VERSION="$(sed -n 's/^set(MR_VERSION "\(.*\)")$/\1/p' "$SRC/CMakeLists.txt")"
MRML_VERSION="$(sed -n 's/^set(MRML_VERSION "\(.*\)")$/\1/p' "$SRC/ofx/ml/CMakeLists.txt")"
MM_VERSION="$(sed -n 's/^set(MM_VERSION "\(.*\)")$/\1/p' "$SRC/ofx/merge/CMakeLists.txt")"
[ -n "$MR_VERSION" ] && [ -n "$MRML_VERSION" ] && [ -n "$MM_VERSION" ] \
  || { echo "error: versions not found in the CMake files" >&2; exit 1; }
echo "Matte Refine $MR_VERSION, Matte Refine ML $MRML_VERSION, Matte Merge $MM_VERSION," \
  "ONNX Runtime $(cat "$ORT_DIR/VERSION_NUMBER")"

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMR_BUILD_ML=ON -DMRML_BUNDLE_ORT=ON \
  -DMR_BUILD_MERGE=ON -DORT_DIR="$ORT_DIR" ${MR_GIT:+-DMR_GIT=$MR_GIT} >/dev/null
cmake --build "$BUILD" -j 8
(cd "$BUILD" && ctest --output-on-failure)

# The runner must find ONNX Runtime inside the bundle only.
RUNNER="$BUILD/MatteRefineML.ofx.bundle/Contents/Resources/mr_ml_runner"
RP="$(readelf -d "$RUNNER" | sed -n 's/.*(\(RUNPATH\|RPATH\)).*\[\(.*\)\]/\2/p')"  # Rocky 9's linker writes DT_RPATH, others DT_RUNPATH
[ "$RP" = '$ORIGIN/lib' ] || { echo "error: mr_ml_runner RUNPATH is '$RP', not \$ORIGIN/lib" >&2; exit 1; }
for f in libonnxruntime.so.1 libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
  [ -e "$BUILD/MatteRefineML.ofx.bundle/Contents/Resources/lib/$f" ] || { echo "error: $f not in the ML bundle" >&2; exit 1; }
done

STAGE="$BUILD/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE" "$DIST"
set_version() { sed -i "s/^VERSION=dev /VERSION=$2 /" "$1"; grep -q "^VERSION=$2 " "$1"; }

# All packages: the user guide (it covers all plugins) with the pictures it and the README show.
add_guide() {
  mkdir -p "$1/docs/images"
  cp "$SRC/README.md" "$SRC/LICENSE" "$SRC/USERGUIDE.md" "$1/"
  for img in $(grep -o 'docs/images/[A-Za-z0-9_.-]*\.png' "$SRC/USERGUIDE.md" "$SRC/README.md" | cut -d: -f2 | sort -u); do
    cp "$SRC/$img" "$1/docs/images/"
  done
}

# Matte Refine (plugin 1).
P1="MatteRefine-$MR_VERSION-linux-x86_64"
mkdir -p "$STAGE/$P1"
cp -R "$BUILD/MatteRefine.ofx.bundle" "$STAGE/$P1/"
cp "$SRC/packaging/install.sh" "$STAGE/$P1/"
add_guide "$STAGE/$P1"
set_version "$STAGE/$P1/install.sh" "$MR_VERSION"

# Matte Refine ML: the bundle (runner, ONNX Runtime in Contents/Resources/lib, licences.txt: the
# gate's reference, gate.h), install.sh, setup.sh, licences.txt, ONNX Runtime's notices, the guide.
P2="MatteRefineML-$MRML_VERSION-linux-x86_64"
mkdir -p "$STAGE/$P2/THIRD_PARTY"
cp -R "$BUILD/MatteRefineML.ofx.bundle" "$STAGE/$P2/"
cp "$SRC/packaging/licences.txt" "$STAGE/$P2/MatteRefineML.ofx.bundle/Contents/Resources/"
cp "$SRC/packaging/install.sh" "$SRC/packaging/setup.sh" "$SRC/packaging/licences.txt" "$STAGE/$P2/"
add_guide "$STAGE/$P2"
cp "$ORT_DIR/LICENSE" "$STAGE/$P2/THIRD_PARTY/onnxruntime-LICENSE.txt"
cp "$ORT_DIR/ThirdPartyNotices.txt" "$STAGE/$P2/THIRD_PARTY/onnxruntime-ThirdPartyNotices.txt"
set_version "$STAGE/$P2/install.sh" "$MRML_VERSION"

# Matte Merge (plugin 3).
P3="MatteMerge-$MM_VERSION-linux-x86_64"
mkdir -p "$STAGE/$P3"
cp -R "$BUILD/MatteMerge.ofx.bundle" "$STAGE/$P3/"
cp "$SRC/packaging/install.sh" "$STAGE/$P3/"
add_guide "$STAGE/$P3"
set_version "$STAGE/$P3/install.sh" "$MM_VERSION"
chmod a+rx "$STAGE/$P1/install.sh" "$STAGE/$P2/install.sh" "$STAGE/$P2/setup.sh" "$STAGE/$P3/install.sh"

for p in "$P1" "$P2" "$P3"; do
  rm -f "$DIST/$p.tar.gz"
  tar -C "$STAGE" --owner=0 --group=0 -czf "$DIST/$p.tar.gz" "$p"
done
(cd "$DIST" && sha256sum "$P1.tar.gz" "$P2.tar.gz" "$P3.tar.gz" >SHA256SUMS)
echo
echo "Packages in $DIST:"
(cd "$DIST" && ls -l "$P1.tar.gz" "$P2.tar.gz" "$P3.tar.gz" SHA256SUMS && cat SHA256SUMS)
echo "Unpacked: $(du -sh "$STAGE/$P1" | cut -f1) (Matte Refine), $(du -sh "$STAGE/$P2" | cut -f1) (Matte Refine ML)," \
  "$(du -sh "$STAGE/$P3" | cut -f1) (Matte Merge)"

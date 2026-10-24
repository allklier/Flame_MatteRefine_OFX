#!/usr/bin/env bash
# Release packages for macOS (Apple Silicon), run from the Linux workstation like scripts/build_mac.sh:
#   HOST=<ssh host> REMOTE=<folder on the Mac> [CMAKE=<cmake on the Mac>] [CMAKE_ARGS=...] [DIST=<local folder>] \
#     scripts/package_release_mac.sh
# Syncs the source (docs included) to REMOTE/release_src, configures a clean REMOTE/release_build (Release,
# MR_BUILD_MERGE=ON; never REMOTE/build, which serves the dev symlinks), builds, runs ctest, checks the
# bundles (arm64 only, minos 15.0, only system libraries, no build paths), stages in a private temporary
# folder on the Mac's own disk (the SMB build share ignores chmod: everything would be 0700), writes into
# REMOTE/release_build/dist, then copies to DIST (default dist/):
#   MatteRefine-<MR_VERSION>-macos-arm64.tar.gz   Matte Refine (plugin 1), install.sh, docs
#   MatteMerge-<MM_VERSION>-macos-arm64.tar.gz    Matte Merge (plugin 3), install.sh, docs
#   SHA256SUMS (on the Mac these two; in DIST merged with the lines already there)
# Matte Refine ML is Linux only (decision 33). Not codesigned or notarised (only the linker's ad-hoc
# signature): install.sh removes the quarantine flag. Writes only under REMOTE and the temporary staging
# folder (removed on exit); never installs into /Library/OFX.
# The script runs itself on the Mac with --on-mac (bash 3.2 there: no arrays under set -u).
set -euo pipefail

if [ "${1:-}" != --on-mac ]; then
  HOST=${HOST:?set HOST to the ssh name of the Mac}
  REMOTE=${REMOTE:?set REMOTE to the build folder on the Mac}
  CMAKE=${CMAKE:-cmake}
  CMAKE_ARGS=${CMAKE_ARGS:-}
  SRC="$(cd "$(dirname "$0")/.." && pwd)"
  DIST="${DIST:-$SRC/dist}"
  ssh -n "$HOST" "mkdir -p '$REMOTE/release_src'"
  rsync -a --delete --exclude .git --exclude build --exclude dist --exclude CLAUDE.local.md \
    --exclude ofx/merge/data "$SRC/" "$HOST:$REMOTE/release_src/"
  # The copy has no .git: name the build after this checkout (Version label, log snapshot).
  GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo nogit)
  [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && GIT="$GIT-dirty"
  GIT=${MR_GIT:-$GIT}
  # CMAKE_ARGS is parsed by the Mac's shell, as in build_mac.sh (quote values with spaces inside it).
  ssh -n "$HOST" "bash '$REMOTE/release_src/scripts/package_release_mac.sh' --on-mac '$REMOTE' '$CMAKE' '$GIT' $CMAKE_ARGS"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  rsync -rt --chmod=D755,F644 "$HOST:$REMOTE/release_build/dist/" "$TMP/"  # the share reports 0700
  (cd "$TMP" && sha256sum -c SHA256SUMS)
  mkdir -p "$DIST"
  cp "$TMP"/*.tar.gz "$DIST/"
  # One SHA256SUMS for the release: these lines replace earlier ones of the same names, others (the
  # Linux tarballs from scripts/package_release.sh) are kept.
  touch "$DIST/SHA256SUMS"
  { awk '{print "  " $2}' "$TMP/SHA256SUMS" | grep -v -F -f - "$DIST/SHA256SUMS" || true; cat "$TMP/SHA256SUMS"; } \
    >"$TMP/merged"
  mv "$TMP/merged" "$DIST/SHA256SUMS"
  (cd "$DIST" && sha256sum -c SHA256SUMS)
  echo "Copied to $DIST"
  exit 0
fi

# ---- on the Mac ----
REMOTE="$2"; CMAKE="$3"; GIT="$4"; shift 4  # the rest: extra configure options
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "error: --on-mac runs on an Apple Silicon Mac" >&2; exit 1; }
case "$CMAKE" in */*) CTEST="${CMAKE%/*}/ctest" ;; *) CTEST=ctest ;; esac
SRC="$REMOTE/release_src"
BUILD="$REMOTE/release_build"
DIST="$BUILD/dist"
export COPYFILE_DISABLE=1  # no AppleDouble ._ files in cp / tar (the build area is an SMB share)

MR_VERSION="$(sed -n 's/^set(MR_VERSION "\(.*\)")$/\1/p' "$SRC/CMakeLists.txt")"
MM_VERSION="$(sed -n 's/^set(MM_VERSION "\(.*\)")$/\1/p' "$SRC/ofx/merge/CMakeLists.txt")"
[ -n "$MR_VERSION" ] && [ -n "$MM_VERSION" ] || { echo "error: versions not found in the CMake files" >&2; exit 1; }
echo "Matte Refine $MR_VERSION, Matte Merge $MM_VERSION, build $GIT, macOS $(sw_vers -productVersion)"

rm -rf "$BUILD"
"$CMAKE" -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMR_BUILD_MERGE=ON -DMR_GIT="$GIT" "$@" >/dev/null
"$CMAKE" --build "$BUILD" -j 8 >"$BUILD/build.log" 2>&1 || { tail -40 "$BUILD/build.log"; exit 1; }
grep -E 'warning' "$BUILD/build.log" || true
(cd "$BUILD" && "$CTEST" --output-on-failure)

# Each plugin binary: arm64 only, loads on macOS 15.0 and later, links only system libraries, holds no
# path of this machine.
check_bin() {
  local f="$1" archs minos libs
  [ -f "$f" ] || { echo "error: $f not built" >&2; exit 1; }
  archs="$(lipo -archs "$f")"
  [ "$archs" = arm64 ] || { echo "error: $f architectures '$archs', not arm64" >&2; exit 1; }
  minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/ {b=1} b && $1 == "minos" {print $2; exit}')"
  [ "$minos" = 15.0 ] || { echo "error: $f minos '$minos', not 15.0" >&2; exit 1; }
  libs="$(otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -v -e '^/usr/lib/' -e '^/System/' || true)"
  [ -z "$libs" ] || { echo "error: $f links non-system libraries: $libs" >&2; exit 1; }
  if strings -a "$f" | grep -F -e "$REMOTE" -e "$HOME" -e /Volumes/; then
    echo "error: $f contains a path of the build machine (above)" >&2; exit 1
  fi
  plutil -lint "${f%/Contents/*}/Contents/Info.plist" >/dev/null
  echo "ok: ${f##*/} $archs minos $minos, $(codesign -dv "$f" 2>&1 | grep -E '^Signature=' || echo 'Signature=none')"
}
check_bin "$BUILD/MatteRefine.ofx.bundle/Contents/MacOS/MatteRefine.ofx"
check_bin "$BUILD/MatteMerge.ofx.bundle/Contents/MacOS/MatteMerge.ofx"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$DIST"
set_version() { sed "s/^VERSION=dev /VERSION=$2 /" "$1" >"$1.tmp" && mv "$1.tmp" "$1"; grep -q "^VERSION=$2 " "$1"; }
# The user guide (it covers all plugins) with the pictures it and the README show.
add_guide() {
  local img
  mkdir -p "$1/docs/images"
  cp "$SRC/README.md" "$SRC/LICENSE" "$SRC/USERGUIDE.md" "$1/"
  for img in $(grep -o 'docs/images/[A-Za-z0-9_.-]*\.png' "$SRC/USERGUIDE.md" "$SRC/README.md" | cut -d: -f2 | sort -u); do
    cp "$SRC/$img" "$1/docs/images/"
  done
}
stage() {  # <package name> <bundle name> <version>
  mkdir -p "$STAGE/$1"
  cp -R "$BUILD/$2.ofx.bundle" "$STAGE/$1/"
  cp "$SRC/packaging/install.sh" "$STAGE/$1/"
  add_guide "$STAGE/$1"
  set_version "$STAGE/$1/install.sh" "$3"
  find "$STAGE/$1" -type d -exec chmod 755 {} +  # the share gave every file 0700 with x set
  find "$STAGE/$1" -type f -exec chmod 644 {} +
  chmod 755 "$STAGE/$1/install.sh" "$STAGE/$1/$2.ofx.bundle/Contents/MacOS/$2.ofx"
  rm -f "$DIST/$1.tar.gz"
  tar -C "$STAGE" --no-mac-metadata --uid 0 --gid 0 --uname root --gname wheel -czf "$DIST/$1.tar.gz" "$1"
  ! tar -tzf "$DIST/$1.tar.gz" | grep -E '(^|/)\._' || { echo "error: AppleDouble files in $1.tar.gz" >&2; exit 1; }
}
P1="MatteRefine-$MR_VERSION-macos-arm64"
P3="MatteMerge-$MM_VERSION-macos-arm64"
stage "$P1" MatteRefine "$MR_VERSION"
stage "$P3" MatteMerge "$MM_VERSION"
(cd "$DIST" && shasum -a 256 "$P1.tar.gz" "$P3.tar.gz" >SHA256SUMS)
echo
echo "Packages in $DIST:"
(cd "$DIST" && ls -l "$P1.tar.gz" "$P3.tar.gz" SHA256SUMS && cat SHA256SUMS)

#!/usr/bin/env bash
# Sync the source to a Mac (Apple Silicon, Command Line Tools), build and test there (as LTX
# Alpha Gen's scripts/build_mac.sh).
#   HOST=<ssh host> REMOTE=<folder on the Mac> [CMAKE=<cmake on the Mac>] scripts/build_mac.sh
# CMAKE_ARGS passes extra configure options (e.g. -DCMAKE_OSX_SYSROOT=<SDK> where a leftover
# SDK breaks the default one). Writes only under $REMOTE (src/ and build/); never installs
# into /Library/OFX. The dev install is a symlink, made by hand with sudo:
#   sudo ln -s <REMOTE>/build/MatteRefine.ofx.bundle /Library/OFX/Plugins/
set -euo pipefail
HOST=${HOST:?set HOST to the ssh name of the Mac}
REMOTE=${REMOTE:?set REMOTE to the build folder on the Mac}
CMAKE=${CMAKE:-cmake}
if [[ $CMAKE == */* ]]; then CTEST=${CTEST:-${CMAKE%/*}/ctest}; else CTEST=${CTEST:-ctest}; fi
CMAKE_ARGS=${CMAKE_ARGS:-}
SRC="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" "mkdir -p '$REMOTE/src'"
rsync -a --delete --exclude .git --exclude build --exclude dist --exclude docs \
  --exclude CLAUDE.local.md "$SRC/" "$HOST:$REMOTE/src/"
# The copy has no .git: name the build after this checkout (Version label, log snapshot).
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo nogit)
[ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && GIT="$GIT-dirty"
ssh "$HOST" "set -e; cd '$REMOTE' && '$CMAKE' -S src -B build -DCMAKE_BUILD_TYPE=Release -DMR_GIT=$GIT $CMAKE_ARGS >/dev/null \
  && '$CMAKE' --build build -j 8 2>&1 | grep -E 'error|warning' || true; \
  cd build && '$CTEST' --output-on-failure && ./test_fill_holes | grep UHD && ./test_solid_core | grep UHD"

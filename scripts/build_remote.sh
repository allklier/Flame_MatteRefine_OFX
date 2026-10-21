#!/usr/bin/env bash
# Sync the source to a Rocky 9 build machine, build and test there (as LTX Alpha Gen).
#   HOST=<ssh host> scripts/build_remote.sh
# Builds in the user's home only; never touches /usr/OFX/Plugins. The one-time install
# symlink is done by hand with sudo:
#   sudo ln -s ~/MatteRefine_build/build/MatteRefine.ofx.bundle /usr/OFX/Plugins/
set -euo pipefail
HOST=${HOST:?set HOST to the ssh name of the Rocky 9 build machine}
REMOTE=${REMOTE:-MatteRefine_build}
SRC="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$HOST" "mkdir -p $REMOTE/src"
rsync -a --delete --exclude .git --exclude build --exclude dist --exclude docs \
  --exclude CLAUDE.local.md "$SRC/" "$HOST:$REMOTE/src/"
# The copy has no .git: name the build after this checkout (Version label, log snapshot).
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo nogit)
[ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && GIT="$GIT-dirty"
ssh "$HOST" "set -e; cd $REMOTE && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DMR_GIT=$GIT >/dev/null \
  && cmake --build build -j 8 2>&1 | grep -E 'error|warning' || true; \
  cd build && ctest --output-on-failure && ./test_fill_holes | grep UHD"

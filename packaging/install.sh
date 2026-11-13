#!/usr/bin/env bash
# Matte Refine for Flame: install or uninstall the plugins.
#   sudo ./install.sh                          install every bundle next to this script
#                                              (MatteRefine.ofx.bundle, MatteRefineML.ofx.bundle,
#                                              MatteMerge.ofx.bundle, EdgeRecipes.ofx.bundle), replacing
#                                              an earlier version; other installed bundles are kept
#   sudo ./install.sh --uninstall              remove the bundles this package ships (those next to it)
#   sudo ./install.sh --plugins-dir DIR [...]  another plugin folder (tests, unusual layouts)
# Installs into /usr/OFX/Plugins (macOS: /Library/OFX/Plugins). Per-user settings, logs and the
# models (~/.config/matte-refine, ~/.cache/matte-refine, ~/.local/share/matte-refine-ml) are not
# touched. Matte Refine ML and Edge Recipes' model steps then need ./setup.sh, run by each artist as
# their normal user (one model folder and one licence acceptance for both).
# Test-only (not for users): MRML_INSTALL_NO_ROOT_CHECK=1 skips the root check.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGINS=/usr/OFX/Plugins
[ "$(uname -s)" = Darwin ] && PLUGINS=/Library/OFX/Plugins  # where the Mac's OFX plugins live
BUNDLES="MatteRefine.ofx.bundle MatteRefineML.ofx.bundle MatteMerge.ofx.bundle EdgeRecipes.ofx.bundle"
VERSION=dev  # set by scripts/package_release.sh

die() { echo "error: $*" >&2; exit 1; }

UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall) UNINSTALL=1 ;;
    --plugins-dir) [ $# -ge 2 ] || die "--plugins-dir needs a folder"; PLUGINS="$2"; shift ;;
    *) die "unknown option $1 (--uninstall, --plugins-dir DIR)" ;;
  esac
  shift
done

[ "$(id -u)" -eq 0 ] || [ "${MRML_INSTALL_NO_ROOT_CHECK:-}" = 1 ] || die "run with sudo (it writes to $PLUGINS)"
if pgrep -x flame >/dev/null 2>&1; then
  [ "$UNINSTALL" = 1 ] && what="NOT UNINSTALLED" || what="NOT INSTALLED"
  echo "$what: Flame is running. Quit Flame, then run this again. Nothing was changed." >&2
  exit 1
fi

# A bundle's runner (Matte Refine ML and Edge Recipes each have their own, Contents/Resources/mr_ml_runner;
# one per user, started by the plugin) keeps running after Flame quits until it is idle for 5 minutes: ask
# the invoking user's runner of that bundle to quit before its files are replaced or removed. Its socket
# is in $XDG_RUNTIME_DIR when Flame had one, else in /tmp: both are tried. Failures are ignored (none
# running, or another user's: those exit on their own).
stop_runner() {  # stop_runner BUNDLE
  local runner="$PLUGINS/$1/Contents/Resources/mr_ml_runner" user="${SUDO_USER:-}" as=()
  [ -x "$runner" ] || return 0
  [ -n "$user" ] && [ "$user" != root ] && as=(sudo -H -u "$user") || user="$(id -un)"
  "${as[@]}" env XDG_RUNTIME_DIR="/run/user/$(id -u "$user")" "$runner" --quit >/dev/null 2>&1 || true
  "${as[@]}" env -u XDG_RUNTIME_DIR "$runner" --quit >/dev/null 2>&1 || true
}

# The bundles of this package: those next to this script. Install and uninstall touch only these, so the
# packages (each plugin alone, or Edge Recipes) can be installed and removed independently.
OURS=""
for b in $BUNDLES; do [ -d "$HERE/$b" ] && OURS="$OURS $b"; done
[ -n "$OURS" ] || die "no plugin bundle ($BUNDLES) found next to this script"

if [ "$UNINSTALL" = 1 ]; then
  for b in $OURS; do
    if [ -e "$PLUGINS/$b" ] || [ -L "$PLUGINS/$b" ]; then
      stop_runner "$b"
      rm -rf "${PLUGINS:?}/$b"
      echo "removed $PLUGINS/$b"
    else
      echo "not installed: $PLUGINS/$b"
    fi
  done
  echo "Done. Your settings (~/.config/matte-refine) and logs (~/.cache/matte-refine) are kept."
  echo "The models and libraries in each user's ~/.local/share/matte-refine-ml are kept (Matte Refine ML and"
  echo "Edge Recipes share them): each user removes them with ./setup.sh --remove."
  exit 0
fi

mkdir -p "$PLUGINS"
for b in $OURS; do
  if [ -e "$PLUGINS/$b" ] || [ -L "$PLUGINS/$b" ]; then
    stop_runner "$b"
    if [ -L "$PLUGINS/$b" ]; then
      echo "replacing the link $PLUGINS/$b -> $(readlink "$PLUGINS/$b")"
    else
      echo "replacing $PLUGINS/$b"
    fi
    rm -rf "${PLUGINS:?}/$b"
  fi
  cp -R "$HERE/$b" "$PLUGINS/"
  chmod -R a+rX "$PLUGINS/$b"
  # macOS: a downloaded tarball marks every file as quarantined; the bundle is not notarised.
  [ "$(uname -s)" = Darwin ] && xattr -dr com.apple.quarantine "$PLUGINS/$b" 2>/dev/null || true
  echo "installed $PLUGINS/$b"
done
if [ -d "$HERE/EdgeRecipes.ofx.bundle" ]; then where="Edge Recipes"; else where="Matte Refine"; fi
echo "Done: version $VERSION installed in $PLUGINS. Start Flame; the nodes are under $where."
if [ -f "$HERE/setup.sh" ]; then
  echo
  echo "Next, as your normal user (not root), each artist who uses Matte Refine ML or Edge Recipes' model steps"
  echo "runs once (a user who already ran it for the other plugin needs nothing more):"
  echo "  $HERE/setup.sh"
  echo "(licences, then the models and CUDA libraries into their home folder, about 1.9 GB)."
fi

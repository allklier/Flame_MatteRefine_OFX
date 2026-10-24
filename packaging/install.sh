#!/usr/bin/env bash
# Matte Refine for Flame: install or uninstall the plugins.
#   sudo ./install.sh                          install every bundle next to this script
#                                              (MatteRefine.ofx.bundle, MatteRefineML.ofx.bundle,
#                                              MatteMerge.ofx.bundle), replacing an earlier version
#   sudo ./install.sh --uninstall              remove all three plugins
#   sudo ./install.sh --plugins-dir DIR [...]  another plugin folder (tests, unusual layouts)
# Installs into /usr/OFX/Plugins (macOS: /Library/OFX/Plugins). Per-user settings, logs and Matte
# Refine ML's models (~/.config/matte-refine, ~/.cache/matte-refine, ~/.local/share/matte-refine-ml)
# are not touched. Matte Refine ML then needs ./setup.sh, run by each artist as their normal user.
# Test-only (not for users): MRML_INSTALL_NO_ROOT_CHECK=1 skips the root check.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGINS=/usr/OFX/Plugins
[ "$(uname -s)" = Darwin ] && PLUGINS=/Library/OFX/Plugins  # where the Mac's OFX plugins live
BUNDLES="MatteRefine.ofx.bundle MatteRefineML.ofx.bundle MatteMerge.ofx.bundle"
ML=MatteRefineML.ofx.bundle
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

# Matte Refine ML's runner (one per user, started by the plugin) keeps running after Flame quits until
# it is idle for 5 minutes: ask the invoking user's runner to quit before its files are replaced. Its
# socket is in $XDG_RUNTIME_DIR when Flame had one, else in /tmp: both are tried. Failures are ignored
# (none running, or another user's: those exit on their own).
stop_runner() {
  local runner="$PLUGINS/$ML/Contents/Resources/mr_ml_runner" user="${SUDO_USER:-}" as=()
  [ -x "$runner" ] || return 0
  [ -n "$user" ] && [ "$user" != root ] && as=(sudo -H -u "$user") || user="$(id -un)"
  "${as[@]}" env XDG_RUNTIME_DIR="/run/user/$(id -u "$user")" "$runner" --quit >/dev/null 2>&1 || true
  "${as[@]}" env -u XDG_RUNTIME_DIR "$runner" --quit >/dev/null 2>&1 || true
}

if [ "$UNINSTALL" = 1 ]; then
  for b in $BUNDLES; do
    if [ -e "$PLUGINS/$b" ] || [ -L "$PLUGINS/$b" ]; then
      [ "$b" = "$ML" ] && stop_runner
      rm -rf "${PLUGINS:?}/$b"
      echo "removed $PLUGINS/$b"
    fi
  done
  echo "Done. Your settings (~/.config/matte-refine) and logs (~/.cache/matte-refine) are kept."
  echo "Matte Refine ML's models and libraries in each user's ~/.local/share/matte-refine-ml are kept:"
  echo "each user removes them with ./setup.sh --remove."
  exit 0
fi

found=0
for b in $BUNDLES; do [ -d "$HERE/$b" ] && found=1; done
[ "$found" = 1 ] || die "no plugin bundle ($BUNDLES) found next to this script"
mkdir -p "$PLUGINS"
for b in $BUNDLES; do
  [ -d "$HERE/$b" ] || continue
  if [ -e "$PLUGINS/$b" ] || [ -L "$PLUGINS/$b" ]; then
    [ "$b" = "$ML" ] && stop_runner
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
echo "Done: version $VERSION installed in $PLUGINS. Start Flame; the nodes are under Matte Refine."
if [ -d "$HERE/$ML" ]; then
  echo
  echo "Next, as your normal user (not root), each artist who uses Matte Refine ML runs once:"
  echo "  $HERE/setup.sh"
  echo "(licences, then the models and CUDA libraries into their home folder, about 1.9 GB)."
fi

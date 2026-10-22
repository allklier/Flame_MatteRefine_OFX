#!/usr/bin/env bash
# Matte Refine for Flame: install or uninstall the plugin.
#   sudo ./install.sh              install (replaces an earlier Matte Refine)
#   sudo ./install.sh --uninstall  remove it
# Installs /usr/OFX/Plugins/MatteRefine.ofx.bundle (macOS: /Library/OFX/Plugins). Per-user settings and logs
# (~/.config/matte-refine, ~/.cache/matte-refine) are not touched.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLUGINS=/usr/OFX/Plugins
[ "$(uname -s)" = Darwin ] && PLUGINS=/Library/OFX/Plugins  # where the Mac's OFX plugins live
BUNDLE=MatteRefine.ofx.bundle
VERSION=dev  # set by scripts/package_release.sh

die() { echo "error: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run with sudo (it writes to $PLUGINS)"
if pgrep -x flame >/dev/null 2>&1; then
  [ "${1:-}" = "--uninstall" ] && what="NOT UNINSTALLED" || what="NOT INSTALLED"
  echo "$what: Flame is running. Quit Flame, then run this again. Nothing was changed." >&2
  exit 1
fi

if [ "${1:-}" = "--uninstall" ]; then
  if [ -e "$PLUGINS/$BUNDLE" ] || [ -L "$PLUGINS/$BUNDLE" ]; then
    rm -rf "${PLUGINS:?}/$BUNDLE"
    echo "removed $PLUGINS/$BUNDLE"
  fi
  echo "Done. Your settings (~/.config/matte-refine) and logs (~/.cache/matte-refine) are kept."
  exit 0
fi

[ -d "$HERE/$BUNDLE" ] || die "$BUNDLE not found next to this script"
mkdir -p "$PLUGINS"
if [ -e "$PLUGINS/$BUNDLE" ] || [ -L "$PLUGINS/$BUNDLE" ]; then
  if [ -L "$PLUGINS/$BUNDLE" ]; then
    echo "replacing the link $PLUGINS/$BUNDLE -> $(readlink "$PLUGINS/$BUNDLE")"
  else
    echo "replacing $PLUGINS/$BUNDLE"
  fi
  rm -rf "${PLUGINS:?}/$BUNDLE"
fi
cp -R "$HERE/$BUNDLE" "$PLUGINS/"
chmod -R a+rX "$PLUGINS/$BUNDLE"
# macOS: a downloaded tarball marks every file as quarantined; the bundle is not notarised.
[ "$(uname -s)" = Darwin ] && xattr -dr com.apple.quarantine "$PLUGINS/$BUNDLE" 2>/dev/null || true
echo "Done: Matte Refine $VERSION installed in $PLUGINS. Start Flame; the node is under Matte Refine > Matte Refine."

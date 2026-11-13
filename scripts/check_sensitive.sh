#!/usr/bin/env bash
# Greps the ADDED lines of the staged diff against .sensitive-patterns (one extended regex per line; the file is
# git-ignored: it names hosts, accounts, clients). Exit 1 on a hit. Known accepted: the plugin id prefix
# of this project (allowlisted below). Usage: scripts/check_sensitive.sh [--all]  (--all scans the whole tree)
set -u
cd "$(git rev-parse --show-toplevel)"
PAT=.sensitive-patterns
[ -f "$PAT" ] || { echo "check_sensitive: no $PAT (git-ignored; create it)"; exit 0; }
ALLOW='com\.jank''lier\.'  # split so this line does not match its own patterns
if [ "${1:-}" = "--all" ]; then
  out=$(git ls-files -z | xargs -0 grep -I -n -E -f "$PAT" 2>/dev/null | grep -v -E "$ALLOW")
else
  out=$(git diff --cached -U0 --no-color | grep -E '^(\+\+\+ |\+[^+])' | grep -E -f "$PAT" | grep -v -E "$ALLOW")
fi
if [ -n "$out" ]; then
  echo "check_sensitive: possible sensitive content in what you are committing:" >&2
  echo "$out" | head -40 >&2
  exit 1
fi

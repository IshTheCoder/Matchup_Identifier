#!/usr/bin/env bash
# Pulls journal-article.tex and refs.bib from a local clone of
# pass-block-drag back into paper/, reversing the path adjustment applied
# by sync_overleaf.sh.
#
# Path convention:
#   Overleaf (pass-block-drag): \input{tables/foo.tex}, \includegraphics{figures/bar.png}
#   Local (paper/paper.tex):    \input{../tables/foo.tex}, \includegraphics{../figures/bar.png}
#
# Usage:
#   ./scripts/sync_from_overleaf.sh              # sync unconditionally
#   ./scripts/sync_from_overleaf.sh --if-changed # only sync if overleaf files changed in HEAD
#
# Config:
#   Set OVERLEAF_REPO env var to override the default path.

set -euo pipefail

OVERLEAF_REPO="${OVERLEAF_REPO:-$HOME/pass-block-drag}"
MAIN_REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PAPER_DIR="$MAIN_REPO/paper"

if [[ ! -d "$OVERLEAF_REPO/.git" ]]; then
  echo "[overleaf-sync] ERROR: $OVERLEAF_REPO is not a git repo."
  echo "  Clone pass-block-drag there first:"
  echo "  git clone <remote-url> $OVERLEAF_REPO"
  exit 1
fi

# --if-changed: only run if the last commit in the overleaf repo touched tex/bib files
if [[ "${1:-}" == "--if-changed" ]]; then
  changed=$(git -C "$OVERLEAF_REPO" diff-tree --no-commit-id -r HEAD --name-only \
    | grep -E '^(journal-article\.tex|refs\.bib)$' || true)
  if [[ -z "$changed" ]]; then
    echo "[overleaf-sync] No tex/bib changes in overleaf HEAD, skipping."
    exit 0
  fi
fi

echo "[overleaf-sync] Pulling from $OVERLEAF_REPO ..."
git -C "$OVERLEAF_REPO" pull --ff-only

# --- journal-article.tex → paper.tex: reverse the path adjustment ---
# Overleaf uses tables/ and figures/; local paper.tex uses ../tables/ and ../figures/
sed 's|{tables/|{../tables/|g; s|{figures/|{../figures/|g' \
  "$OVERLEAF_REPO/journal-article.tex" > "$PAPER_DIR/paper.tex"

# --- refs.bib ---
cp "$OVERLEAF_REPO/refs.bib" "$PAPER_DIR/refs.bib"

# --- commit ---
cd "$MAIN_REPO"
git add paper/paper.tex paper/refs.bib

if git diff --cached --quiet; then
  echo "[overleaf-sync] Nothing changed in paper/."
  exit 0
fi

commit_msg=$(git -C "$OVERLEAF_REPO" log -1 --pretty=format:"sync from overleaf: %s")
git commit -m "$commit_msg"

echo "[overleaf-sync] Done. Review and push when ready."

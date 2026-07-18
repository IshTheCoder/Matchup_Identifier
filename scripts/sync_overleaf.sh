#!/usr/bin/env bash
# Syncs prose (paper/paper.tex → journal-article.tex, paper/refs.bib), generated
# figures (figures/*.png, figures/*.pdf), tables (tables/*.tex), and stylesheets
# to a local clone of pass-block-drag (Overleaf), then commits and pushes.
#
# Path adjustment applied to paper.tex on the way out:
#   ../tables/ → tables/   and   ../figures/ → figures/
#
# Usage:
#   ./scripts/sync_overleaf.sh              # sync unconditionally
#   ./scripts/sync_overleaf.sh --if-changed # only sync if relevant files changed in HEAD
#
# Config:
#   Set OVERLEAF_REPO env var to override the default path.

set -euo pipefail

OVERLEAF_REPO="${OVERLEAF_REPO:-$HOME/pass-block-drag}"
MAIN_REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PAPER_DIR="$MAIN_REPO/paper"

# --if-changed: only run if the last commit touched generated outputs or stylesheets
if [[ "${1:-}" == "--if-changed" ]]; then
  changed=$(git -C "$MAIN_REPO" diff-tree --no-commit-id -r HEAD --name-only \
    | grep -E '^figures/.*\.(png|pdf)$|^tables/.*\.tex$|^paper/[^/]+\.(cls|sty|bst|tex|bib)$' || true)
  if [[ -z "$changed" ]]; then
    echo "[overleaf-sync] No output changes in this commit, skipping."
    exit 0
  fi
fi

if [[ ! -d "$OVERLEAF_REPO/.git" ]]; then
  echo "[overleaf-sync] ERROR: $OVERLEAF_REPO is not a git repo."
  echo "  Clone pass-block-drag there first:"
  echo "  git clone <remote-url> $OVERLEAF_REPO"
  exit 1
fi

echo "[overleaf-sync] Syncing to $OVERLEAF_REPO ..."

# --- paper.tex → journal-article.tex (adjust paths for Overleaf's flat layout) ---
tex_src="$PAPER_DIR/paper.tex"
if [[ ! -f "$tex_src" ]]; then
  echo "[overleaf-sync] ERROR: $tex_src not found." >&2
  exit 1
fi
sed 's|{../tables/|{tables/|g; s|{../figures/|{figures/|g' \
  "$tex_src" > "$OVERLEAF_REPO/journal-article.tex"

# --- refs.bib ---
[[ -f "$PAPER_DIR/refs.bib" ]] && cp "$PAPER_DIR/refs.bib" "$OVERLEAF_REPO/refs.bib"

# --- stylesheets: .cls, .sty, .bst ---
for f in "$PAPER_DIR"/*.{cls,sty,bst}; do
  [[ -f "$f" ]] && cp "$f" "$OVERLEAF_REPO/$(basename "$f")"
done

# --- images and \input-ed .tex files referenced in paper.tex ---
{
  uncommented=$(grep -v '^\s*%' "$tex_src")
  # all \includegraphics paths
  grep -oP '\\includegraphics\[?[^\]]*\]?\{[^}]+\}' <<< "$uncommented" \
    | grep -oP '\{[^}]+\}' | tr -d '{}'
  # \input paths that point into ../tables/ or ../figures/
  grep -oP '\\input\{[^}]+\}' <<< "$uncommented" \
    | grep -oP '\{[^}]+\}' | tr -d '{}' \
    | grep -E '\.\./tables/|\.\./figures/'
} | sort -u | while IFS= read -r file; do
      # file is relative to paper/, e.g. ../tables/foo.tex, ../figures/bar.png, or baz.png
      src="$PAPER_DIR/$file"
      dest_rel="${file#../}"   # strip leading ../ to get the overleaf-relative path
      dest="$OVERLEAF_REPO/$dest_rel"

      if [[ -f "$src" ]]; then
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
      else
        echo "[overleaf-sync] WARNING: missing $file"
      fi
    done

# --- commit and push ---
cd "$OVERLEAF_REPO"
git add -A

if git diff --cached --quiet; then
  echo "[overleaf-sync] Nothing changed in overleaf repo."
  exit 0
fi

commit_msg=$(git -C "$MAIN_REPO" log -1 --pretty=format:"sync: %s")
git commit -m "$commit_msg"
git push origin main

echo "[overleaf-sync] Done."

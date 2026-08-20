#!/usr/bin/env bash
# Uploads a fine-tuned GGUF to a Hugging Face Hub repo -- the only reason
# to do this at all is to make the file reachable by something that can't
# see this machine's local filesystem, specifically
# .github/workflows/publish-model.yml: a GitHub-hosted Actions runner is a
# fresh cloud VM every run, so it can only *download* the GGUF from
# somewhere on the network, never read it directly off this machine. If
# publishing straight from here instead, this script isn't needed at all
# -- use scripts/push_model.sh directly.
#
# Usage:
#   ./scripts/push_to_huggingface.sh <path-to-model.gguf> <hf-username>/<repo-name> [path-in-repo]
set -euo pipefail

GGUF_PATH="${1:?usage: $0 <path-to-model.gguf> <hf-username>/<repo-name> [path-in-repo]}"
REPO_ID="${2:?usage: $0 <path-to-model.gguf> <hf-username>/<repo-name> [path-in-repo]}"
PATH_IN_REPO="${3:-$(basename "$GGUF_PATH")}"

if [ ! -f "$GGUF_PATH" ]; then
  echo "error: $GGUF_PATH not found" >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "error: not logged in to Hugging Face -- run: hf auth login" >&2
  exit 1
fi

echo "uploading $GGUF_PATH to $REPO_ID as $PATH_IN_REPO..."
hf upload "$REPO_ID" "$GGUF_PATH" "$PATH_IN_REPO"

echo ""
echo "done. Trigger .github/workflows/publish-model.yml with:"
echo "  hf_repo: $REPO_ID"
echo "  gguf_filename: $PATH_IN_REPO"

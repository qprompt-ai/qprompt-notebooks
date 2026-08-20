#!/usr/bin/env bash
# Packages a fine-tuned GGUF model as an OCI artifact and pushes it to a
# container registry, so it can be pulled by Docker Compose's models.
#
# `docker model package`/`push` are registry-agnostic -- this was always
# true, the script just used to only show a Docker Hub example. GHCR
# (ghcr.io) is the recommended target for an org namespace like
# qprompt/...: GitHub orgs are free to create and GHCR hosts public
# packages at no cost, unlike a real Docker Hub Organization, which
# requires a paid Team plan. A bare `user/repo:tag` (no registry host)
# still defaults to Docker Hub, same as it always did.
#
# Usage:
#   ./scripts/push_model.sh <path-to-model.gguf> ghcr.io/qprompt/<repo>:<tag> [license-file]
#   ./scripts/push_model.sh <path-to-model.gguf> <dockerhub-user>/<repo>:<tag> [license-file]
set -euo pipefail

GGUF_PATH="${1:?usage: $0 <path-to-model.gguf> <registry>/<repo>:<tag> [license-file]}"
TAG="${2:?usage: $0 <path-to-model.gguf> <registry>/<repo>:<tag> [license-file]}"
LICENSE_PATH="${3:-}"

if [ ! -f "$GGUF_PATH" ]; then
  echo "error: $GGUF_PATH not found" >&2
  exit 1
fi

if ! docker model version >/dev/null 2>&1; then
  echo "error: Docker Model Runner plugin not available (docker model version failed)" >&2
  echo "install/enable it first: https://docs.docker.com/ai/model-runner/" >&2
  exit 1
fi

REGISTRY="${TAG%%/*}"
if [ "$REGISTRY" = "$TAG" ] || [[ "$REGISTRY" != *.* ]]; then
  # No registry host in the tag (e.g. "you/repo:tag") -- defaults to Docker Hub.
  REGISTRY="docker.io"
fi

echo "logging in to $REGISTRY (skip if already logged in)..."
docker login "$REGISTRY"

PACKAGE_ARGS=(--gguf "$GGUF_PATH")
if [ -n "$LICENSE_PATH" ]; then
  PACKAGE_ARGS+=(--license "$LICENSE_PATH")
fi

echo "packaging $GGUF_PATH as $TAG..."
docker model package "${PACKAGE_ARGS[@]}" "$TAG"

echo "pushing $TAG..."
docker model push "$TAG"

echo "done."

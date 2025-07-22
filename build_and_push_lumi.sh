#!/usr/bin/env bash
# build_and_push_lumi.sh
#
# Convenience script to build the ROCm-enabled Lumi evaluation environment
# locally and push it to GitHub Container Registry (GHCR).
#
# Prerequisites:
#   • Docker (with BuildKit) is installed and the current user can run it
#   • A .env file exists in the project root containing at least:
#       GHCR_USERNAME=<your-github-username>
#       GHCR_TOKEN=<a GitHub PAT with `packages:write` scope>
# Usage:
#   ./build_and_push_lumi.sh [TAG]
#
# If TAG is omitted, the short git commit SHA is used. The script always also
# pushes the image with the `latest` tag.

set -euo pipefail

# ----------------- Load credentials from .env ------------------------------
if [[ -f .env ]]; then
  # shellcheck disable=SC2163
  export $(grep -v '^#' .env | xargs -0 2>/dev/null || grep -v '^#' .env | xargs)
fi

if [[ -z "${GHCR_USERNAME:-}" || -z "${GHCR_TOKEN:-}" ]]; then
  echo "[ERROR] GHCR_USERNAME or GHCR_TOKEN is not set.\n"
  echo "Create a .env file with:\n  GHCR_USERNAME=<your_username>\n  GHCR_TOKEN=<your_pat>\n"
  exit 1
fi

# ----------------- Determine image reference -------------------------------
IMAGE="ghcr.io/openeurollm/eval_env-lumi"
TAG=${1:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}

# ----------------- Docker login -------------------------------------------
echo "[INFO] Logging in to ghcr.io as $GHCR_USERNAME"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin

# ----------------- Build image --------------------------------------------
echo "[INFO] Building Docker image $IMAGE with tags: latest, $TAG"
docker build \
  --progress=auto \
  --platform linux/amd64 \
  -f docker/lumi.Dockerfile \
  -t "$IMAGE:latest" \
  -t "$IMAGE:$TAG" \
  --label org.opencontainers.image.source=https://github.com/OpenEuroLLM/multi-cluster-eval \
  --no-cache \
  .

# ----------------- Push image ---------------------------------------------
echo "[INFO] Pushing tags to GHCR"
docker push "$IMAGE:latest"
docker push "$IMAGE:$TAG"

echo "[SUCCESS] Image available as:"
echo "  $IMAGE:latest"
echo "  $IMAGE:$TAG" 
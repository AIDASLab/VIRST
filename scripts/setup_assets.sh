#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
VIDEOCHAT_REPO="${VIDEOCHAT_REPO:-https://github.com/OpenGVLab/VideoChat-Flash.git}"
VIDEOCHAT_MODEL="${VIDEOCHAT_MODEL:-OpenGVLab/VideoChat-Flash-Qwen2-7B_res448}"
VIRST_DRIVE_ID="${VIRST_DRIVE_ID:-19PrTMWWzGHBTrZ0JTe1feH205vjHkoNx}"
SAM2_URL="${SAM2_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"

mkdir -p "${CHECKPOINT_DIR}" "${THIRD_PARTY_DIR}"

if [[ ! -d "${THIRD_PARTY_DIR}/VideoChat-Flash/.git" ]]; then
    git clone "${VIDEOCHAT_REPO}" "${THIRD_PARTY_DIR}/VideoChat-Flash"
else
    echo "VideoChat-Flash source already exists; skipping clone."
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli is missing. Install requirements.txt first." >&2
    exit 1
fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is missing. Install requirements.txt first." >&2
    exit 1
fi

VIDEOCHAT_COMPLETE=true
for required_file in model.safetensors.index.json \
    model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
    model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
    if [[ ! -s "${CHECKPOINT_DIR}/videochat/${required_file}" ]]; then
        VIDEOCHAT_COMPLETE=false
    fi
done

if [[ "${VIDEOCHAT_COMPLETE}" != true ]]; then
    huggingface-cli download "${VIDEOCHAT_MODEL}" \
        --local-dir "${CHECKPOINT_DIR}/videochat"
else
    echo "VideoChat-Flash checkpoint already exists; skipping download."
fi

if [[ ! -f "${CHECKPOINT_DIR}/sam2.1_hiera_large.pt" ]]; then
    curl --fail --location --retry 3 \
        --output "${CHECKPOINT_DIR}/sam2.1_hiera_large.pt" "${SAM2_URL}"
else
    echo "SAM2 checkpoint already exists; skipping download."
fi

if [[ ! -f "${CHECKPOINT_DIR}/virst_checkpoint.pt" ]]; then
    gdown "https://drive.google.com/uc?id=${VIRST_DRIVE_ID}" \
        --output "${CHECKPOINT_DIR}/virst_checkpoint.pt"
else
    echo "VIRST checkpoint already exists; skipping download."
fi

echo "All model assets are available under ${CHECKPOINT_DIR}."

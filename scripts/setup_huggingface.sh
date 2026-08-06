#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

STT_MODEL="${STT_MODEL:-ai4bharat/indic-conformer-600m-multilingual}"
TTS_MODEL="${TTS_MODEL:-ai4bharat/indic-parler-tts}"

usage() {
    printf '%s\n' \
        "Usage: $0 [--packages-only | --models-only]" \
        "" \
        "Installs the Hugging Face, PyTorch, and Parler-TTS packages and" \
        "downloads the voice models used by this project." \
        "" \
        "Environment variables:" \
        "  MODEL_DIR   Download directory (default: <project>/models)" \
        "  PYTHON_BIN  Python used to create the virtual environment" \
        "  VENV_DIR    Virtual environment (default: <project>/.venv)" \
        "  HF_TOKEN    Hugging Face token, if a model requires access" \
        "  STT_MODEL   STT repository ID" \
        "  TTS_MODEL   TTS repository ID"
}

INSTALL_PACKAGES=true
DOWNLOAD_MODELS=true

case "${1:-}" in
    --packages-only) DOWNLOAD_MODELS=false ;;
    --models-only) INSTALL_PACKAGES=false ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'Error: Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    printf 'Creating virtual environment at %s\n' "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

if "${INSTALL_PACKAGES}"; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install \
        'huggingface_hub[cli]>=0.24' \
        'transformers>=4.40' \
        torch \
        torchaudio \
        'git+https://github.com/huggingface/parler-tts.git'
fi

if "${DOWNLOAD_MODELS}"; then
    mkdir -p "${MODEL_DIR}"

    download_model() {
        local repository_id="$1"
        local destination="$2"

        printf 'Downloading %s to %s\n' "${repository_id}" "${destination}"
        "${PYTHON_BIN}" -c \
            'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' \
            "${repository_id}" "${destination}"
    }

    download_model "${STT_MODEL}" "${MODEL_DIR}/indic-conformer-600m-multilingual"
    download_model "${TTS_MODEL}" "${MODEL_DIR}/indic-parler-tts"
fi

printf 'Hugging Face setup completed successfully.\n'

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash accept_asr_mode.sh --dataset DATASET_JSON --outdir OUTDIR [options]

Acceptance focus:
  C2 ASR output mode: onebest for standard C2 output, nbest for C3 correction input.

Key options:
  --dataset PATH          Dataset JSON.
  --outdir PATH           Output directory.
  --model PATH            Whisper model path.
  --asr-mode MODE         onebest | nbest. Default: nbest.
  --vad-backend NAME      energy | silero. Default: silero.
  --start N               Start offset. Default: 0.
  --n N                   Number of items. Default: 1.
  --language CODE         ASR language code. Default: en.
  --nbest N               N-best candidates per chunk. Default: 5.
  --beam-size N           Beam size for nbest mode. Default: 5.
  --diarize 0|1           Enable PyAnnote speaker diarization. Default: 0.
  --offline 0|1           Enable HuggingFace offline mode. Default: 0.
  --help                  Show this help.
  -- ARGS                 Pass remaining args to c2_asr.py.

Examples:
  bash accept_asr_mode.sh --dataset ../data/cremd-test200.json --outdir ../outputs/acceptance/asr_nbest --asr-mode nbest --n 1
  bash accept_asr_mode.sh --dataset ../data/cremd-test200.json --outdir ../outputs/acceptance/asr_onebest --asr-mode onebest --n 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C2_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

DATASET="${DATASET:-${C2_DIR}/data/cremd-test200.json}"
OUTDIR="${OUTDIR:-${C2_DIR}/outputs/acceptance/asr_mode}"
MODEL="${MODEL:-/root/siton-tmp/assignment_C/model/whisper-large-v3}"
ASR_MODE="${ASR_MODE:-nbest}"
VAD_BACKEND="${VAD_BACKEND:-silero}"
START="${START:-0}"
N="${N:-1}"
LANGUAGE="${LANGUAGE:-en}"
NBEST="${NBEST:-5}"
BEAM_SIZE="${BEAM_SIZE:-5}"
DIARIZE="${DIARIZE:-0}"
OFFLINE="${OFFLINE:-0}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --asr-mode) ASR_MODE="$2"; shift 2 ;;
    --vad-backend) VAD_BACKEND="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --nbest) NBEST="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    --diarize) DIARIZE="$2"; shift 2 ;;
    --offline) OFFLINE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_script_logging "accept_asr_mode" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[accept_asr_mode] DATASET=${DATASET}"
echo "[accept_asr_mode] OUTDIR=${OUTDIR}"
echo "[accept_asr_mode] MODEL=${MODEL}"
echo "[accept_asr_mode] ASR_MODE=${ASR_MODE}"
echo "[accept_asr_mode] VAD_BACKEND=${VAD_BACKEND}"
echo "[accept_asr_mode] START=${START} N=${N}"
echo "[accept_asr_mode] DIARIZE=${DIARIZE}"

cmd=(
  python3 "${C2_DIR}/code/c2_asr.py"
  --dataset "${DATASET}"
  --model "${MODEL}"
  --outdir "${OUTDIR}"
  --asr_mode "${ASR_MODE}"
  --vad_backend "${VAD_BACKEND}"
  --start "${START}"
  --n "${N}"
  --language "${LANGUAGE}"
  --nbest "${NBEST}"
  --beam_size "${BEAM_SIZE}"
)

if [[ "${DIARIZE}" == "1" ]]; then
  cmd+=(--diarize)
fi
if [[ "${OFFLINE}" == "1" ]]; then
  cmd+=(--offline)
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" "${cmd[@]}" "${EXTRA_ARGS[@]}"

if [[ "${ASR_MODE}" == "nbest" ]]; then
  echo "[accept_asr_mode] saved: ${OUTDIR}/asr_nbest_predictions.json"
else
  echo "[accept_asr_mode] saved: ${OUTDIR}/asr_predictions.json"
fi
echo "[accept_asr_mode] saved chunks: ${OUTDIR}/stage2_chunks"

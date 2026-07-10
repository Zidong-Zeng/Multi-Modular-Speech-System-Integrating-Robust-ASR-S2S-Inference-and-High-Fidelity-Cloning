#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C2_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

DATASET="${DATASET:-${C2_DIR}/data/cremd-test200.json}"
OUTDIR="${OUTDIR:-${C2_DIR}/outputs/c2_vad_asr/demo}"
MODEL="${MODEL:-/root/siton-tmp/assignment_C/model/whisper-large-v3}"
ASR_MODE="${ASR_MODE:-nbest}"          # onebest | nbest
VAD_BACKEND="${VAD_BACKEND:-silero}"   # energy | silero
ENABLE_PYANNOTE="${ENABLE_PYANNOTE:-0}" # 0 | 1
START="${START:-0}"
N="${N:-0}"
LANGUAGE="${LANGUAGE:-en}"
NBEST="${NBEST:-5}"
BEAM_SIZE="${BEAM_SIZE:-5}"

init_script_logging "c2_vad_asr" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[C2] DATASET=${DATASET}"
echo "[C2] OUTDIR=${OUTDIR}"
echo "[C2] MODEL=${MODEL}"
echo "[C2] ASR_MODE=${ASR_MODE}"
echo "[C2] VAD_BACKEND=${VAD_BACKEND}"
echo "[C2] ENABLE_PYANNOTE=${ENABLE_PYANNOTE}"
echo "[C2] START=${START}"
echo "[C2] N=${N}"

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

if [[ "${ENABLE_PYANNOTE}" == "1" ]]; then
  cmd+=(--diarize)
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" "${cmd[@]}" "$@"

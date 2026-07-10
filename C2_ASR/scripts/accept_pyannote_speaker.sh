#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash accept_pyannote_speaker.sh --dataset DATASET_JSON --outdir OUTDIR [options]

Acceptance focus:
  PyAnnote speaker diarization in C2 and speaker-level transcript extraction.

Key options:
  --dataset PATH          Dataset JSON.
  --outdir PATH           Output directory.
  --model PATH            Whisper model path.
  --vad-backend NAME      energy | silero. Default: silero.
  --start N               Start offset. Default: 0.
  --n N                   Number of items. Default: 1.
  --language CODE         ASR language code. Default: en.
  --asr-mode MODE         onebest | nbest. Default: onebest.
  --pyannote-model PATH   Local pyannote speaker-diarization model/config.
  --pyannote-segmentation-model PATH
                          Local pyannote segmentation model.
  --hf-token TOKEN        Optional HuggingFace token if remote pyannote is used.
  --report-mode MODE      transcript | script. Default: script.
  --help                  Show this help.
  -- ARGS                 Pass remaining args to c2_asr.py.

Example:
  bash accept_pyannote_speaker.sh --dataset ../data/cremd-test200.json --outdir ../outputs/acceptance/pyannote --n 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C2_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

DATASET="${DATASET:-${C2_DIR}/data/cremd-test200.json}"
OUTDIR="${OUTDIR:-${C2_DIR}/outputs/acceptance/pyannote_speaker}"
MODEL="${MODEL:-/root/siton-tmp/assignment_C/model/whisper-large-v3}"
ASR_MODE="${ASR_MODE:-onebest}"
VAD_BACKEND="${VAD_BACKEND:-silero}"
START="${START:-0}"
N="${N:-1}"
LANGUAGE="${LANGUAGE:-en}"
PYANNOTE_MODEL="${PYANNOTE_MODEL:-/root/siton-tmp/assignment_C/model/pyannote-speaker-diarization-3.1}"
PYANNOTE_SEGMENTATION_MODEL="${PYANNOTE_SEGMENTATION_MODEL:-/root/siton-tmp/assignment_C/model/pyannote-segmentation-3.0}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-${HF_TOKEN:-}}"
REPORT_MODE="${REPORT_MODE:-script}"

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
    --pyannote-model) PYANNOTE_MODEL="$2"; shift 2 ;;
    --pyannote-segmentation-model) PYANNOTE_SEGMENTATION_MODEL="$2"; shift 2 ;;
    --hf-token) HF_TOKEN_VALUE="$2"; shift 2 ;;
    --report-mode) REPORT_MODE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_script_logging "accept_pyannote_speaker" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[accept_pyannote_speaker] DATASET=${DATASET}"
echo "[accept_pyannote_speaker] OUTDIR=${OUTDIR}"
echo "[accept_pyannote_speaker] ASR_MODE=${ASR_MODE}"
echo "[accept_pyannote_speaker] VAD_BACKEND=${VAD_BACKEND}"
echo "[accept_pyannote_speaker] PYANNOTE_MODEL=${PYANNOTE_MODEL}"
echo "[accept_pyannote_speaker] START=${START} N=${N}"

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
  --diarize
  --pyannote_model "${PYANNOTE_MODEL}"
  --pyannote_segmentation_model "${PYANNOTE_SEGMENTATION_MODEL}"
)

if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  cmd+=(--hf_token "${HF_TOKEN_VALUE}")
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" "${cmd[@]}" "${EXTRA_ARGS[@]}"

if [[ "${ASR_MODE}" == "nbest" ]]; then
  PRED_JSON="${OUTDIR}/asr_nbest_predictions.json"
else
  PRED_JSON="${OUTDIR}/asr_predictions.json"
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" python3 "${C2_DIR}/code/extract_speaker.py" \
  --input "${PRED_JSON}" \
  --mode "${REPORT_MODE}" \
  --out_json "${OUTDIR}/speaker_transcripts.json" \
  --out_txt "${OUTDIR}/speaker_transcripts.txt" \
  --out_juben "${OUTDIR}/juben.md"

echo "[accept_pyannote_speaker] saved: ${PRED_JSON}"
echo "[accept_pyannote_speaker] saved: ${OUTDIR}/speaker_transcripts.json"
echo "[accept_pyannote_speaker] saved: ${OUTDIR}/speaker_transcripts.txt"
echo "[accept_pyannote_speaker] saved: ${OUTDIR}/juben.md"

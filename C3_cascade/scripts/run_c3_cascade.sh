#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C3_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

C2_JSON="${C2_JSON:-${C3_DIR}/outputs/c2-nbest/cremd-tiny/asr_nbest_predictions.json}"
OUTDIR="${OUTDIR:-${C3_DIR}/outputs/c3-corrected/demo}"
CORRECTION_BACKEND="${CORRECTION_BACKEND:-local}" # local | openai_compatible
CORRECTION_MODEL="${CORRECTION_MODEL:-/root/siton-tmp/assignment_C/model/Qwen3-4B}"
TRANSLATION_MODEL="${TRANSLATION_MODEL:-/root/siton-tmp/assignment_C/model/Qwen2.5-1.5B-Instruct}"
CORRECTION_API_BASE="${CORRECTION_API_BASE:-https://api.deepseek.com/v1}"
CORRECTION_API_MODEL="${CORRECTION_API_MODEL:-deepseek-chat}"
CORRECTION_API_KEY_ENV="${CORRECTION_API_KEY_ENV:-CORRECTION_API_KEY}"

init_script_logging "c3_cascade" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[C3] C2_JSON=${C2_JSON}"
echo "[C3] OUTDIR=${OUTDIR}"
echo "[C3] CORRECTION_BACKEND=${CORRECTION_BACKEND}"
echo "[C3] CORRECTION_MODEL=${CORRECTION_MODEL}"
echo "[C3] TRANSLATION_MODEL=${TRANSLATION_MODEL}"

cmd=(
  python3 -m c3.cli
  --c2_json "${C2_JSON}"
  --outdir "${OUTDIR}"
  --correction_backend "${CORRECTION_BACKEND}"
  --correction_model "${CORRECTION_MODEL}"
  --translation_model "${TRANSLATION_MODEL}"
  --correction_api_base "${CORRECTION_API_BASE}"
  --correction_api_model "${CORRECTION_API_MODEL}"
  --correction_api_key_env "${CORRECTION_API_KEY_ENV}"
)

PYTHONPATH="${C3_DIR}/code:${PYTHONPATH:-}" "${cmd[@]}" "$@"

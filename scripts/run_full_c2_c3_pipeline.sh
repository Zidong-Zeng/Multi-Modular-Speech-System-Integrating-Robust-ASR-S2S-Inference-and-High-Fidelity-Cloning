#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_C_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
C2_DIR="${ASSIGNMENT_C_DIR}/C2_ASR"
C3_DIR="${ASSIGNMENT_C_DIR}/C3_cascade"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

RUN_ID="${RUN_ID:-demo}"
C2_OUTDIR="${C2_OUTDIR:-${C2_DIR}/outputs/c2_vad_asr/${RUN_ID}}"
C3_OUTDIR="${C3_OUTDIR:-${C3_DIR}/outputs/c3-corrected/${RUN_ID}}"

init_script_logging "full_c2_c3_pipeline" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

echo "[full] RUN_ID=${RUN_ID}"
echo "[full] C2_OUTDIR=${C2_OUTDIR}"
echo "[full] C3_OUTDIR=${C3_OUTDIR}"

ASR_MODE="${ASR_MODE:-nbest}" OUTDIR="${C2_OUTDIR}" bash "${C2_DIR}/scripts/run_c2_vad_asr.sh" "$@"

C2_JSON="${C2_OUTDIR}/asr_nbest_predictions.json"
if [[ ! -f "${C2_JSON}" ]]; then
  echo "[full] expected C2 n-best artifact not found: ${C2_JSON}" >&2
  exit 1
fi

C2_JSON="${C2_JSON}" OUTDIR="${C3_OUTDIR}" bash "${C3_DIR}/scripts/run_c3_cascade.sh"

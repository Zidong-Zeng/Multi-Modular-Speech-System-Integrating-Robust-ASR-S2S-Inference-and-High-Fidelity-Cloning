#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash accept_correction_only.sh --c2-json C2_NBEST_JSON --outdir OUTDIR [options]

Acceptance focus:
  C3 correction-unit construction and correction model output, without translation.

Key options:
  --c2-json PATH                  C2 asr_nbest_predictions.json.
  --outdir PATH                   Output directory.
  --correction-backend NAME       local | openai_compatible. Default: local.
  --correction-model PATH         Local HuggingFace correction model path.
  --correction-api-base URL       OpenAI-compatible API base.
  --correction-api-model NAME     OpenAI-compatible correction model name.
  --correction-api-key-env NAME   Env var containing API key. Default: CORRECTION_API_KEY.
  --max-new-tokens N              Correction max_new_tokens. Default: 4096.
  --min-unit-ms N                 Minimum correction unit duration. Default: 6000.
  --min-unit-words N              Minimum correction unit words. Default: 10.
  --allow-cross-speaker 0|1       Allow units to cross speaker boundary. Default: 0.
  --help                          Show this help.

Example:
  bash accept_correction_only.sh --c2-json ../C2_ASR/outputs/acceptance/asr_nbest/asr_nbest_predictions.json --outdir ../outputs/acceptance/correction
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C3_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

C2_JSON="${C2_JSON:-${C3_DIR}/outputs/c2-nbest/cremd-tiny/asr_nbest_predictions.json}"
OUTDIR="${OUTDIR:-${C3_DIR}/outputs/acceptance/correction_only}"
CORRECTION_BACKEND="${CORRECTION_BACKEND:-local}"
CORRECTION_MODEL="${CORRECTION_MODEL:-/root/siton-tmp/assignment_C/model/Qwen3-4B}"
CORRECTION_API_BASE="${CORRECTION_API_BASE:-https://api.deepseek.com/v1}"
CORRECTION_API_MODEL="${CORRECTION_API_MODEL:-deepseek-chat}"
CORRECTION_API_KEY_ENV="${CORRECTION_API_KEY_ENV:-CORRECTION_API_KEY}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MIN_UNIT_MS="${MIN_UNIT_MS:-6000}"
MIN_UNIT_WORDS="${MIN_UNIT_WORDS:-10}"
TARGET_MAX_MS="${TARGET_MAX_MS:-25000}"
HARD_MAX_MS="${HARD_MAX_MS:-29500}"
MAX_UNIT_WORDS="${MAX_UNIT_WORDS:-80}"
MAX_MERGE_GAP_MS="${MAX_MERGE_GAP_MS:-1200}"
FALLBACK_MIN_MS="${FALLBACK_MIN_MS:-0}"
FALLBACK_MIN_WORDS="${FALLBACK_MIN_WORDS:-0}"
MAX_OVERLAP_WORDS="${MAX_OVERLAP_WORDS:-8}"
ALLOW_CROSS_SPEAKER="${ALLOW_CROSS_SPEAKER:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --c2-json) C2_JSON="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --correction-backend) CORRECTION_BACKEND="$2"; shift 2 ;;
    --correction-model) CORRECTION_MODEL="$2"; shift 2 ;;
    --correction-api-base) CORRECTION_API_BASE="$2"; shift 2 ;;
    --correction-api-model) CORRECTION_API_MODEL="$2"; shift 2 ;;
    --correction-api-key-env) CORRECTION_API_KEY_ENV="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --min-unit-ms) MIN_UNIT_MS="$2"; shift 2 ;;
    --min-unit-words) MIN_UNIT_WORDS="$2"; shift 2 ;;
    --target-max-ms) TARGET_MAX_MS="$2"; shift 2 ;;
    --hard-max-ms) HARD_MAX_MS="$2"; shift 2 ;;
    --max-unit-words) MAX_UNIT_WORDS="$2"; shift 2 ;;
    --max-merge-gap-ms) MAX_MERGE_GAP_MS="$2"; shift 2 ;;
    --fallback-min-ms) FALLBACK_MIN_MS="$2"; shift 2 ;;
    --fallback-min-words) FALLBACK_MIN_WORDS="$2"; shift 2 ;;
    --max-overlap-words) MAX_OVERLAP_WORDS="$2"; shift 2 ;;
    --allow-cross-speaker) ALLOW_CROSS_SPEAKER="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_script_logging "accept_correction_only" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[accept_correction_only] C2_JSON=${C2_JSON}"
echo "[accept_correction_only] OUTDIR=${OUTDIR}"
echo "[accept_correction_only] CORRECTION_BACKEND=${CORRECTION_BACKEND}"
echo "[accept_correction_only] CORRECTION_MODEL=${CORRECTION_MODEL}"

PYTHONPATH="${C3_DIR}/code:${PYTHONPATH:-}" python3 - "${C2_JSON}" "${OUTDIR}" \
  "${CORRECTION_BACKEND}" "${CORRECTION_MODEL}" "${CORRECTION_API_BASE}" \
  "${CORRECTION_API_MODEL}" "${CORRECTION_API_KEY_ENV}" "${MAX_NEW_TOKENS}" \
  "${MIN_UNIT_MS}" "${MIN_UNIT_WORDS}" "${TARGET_MAX_MS}" "${HARD_MAX_MS}" \
  "${MAX_UNIT_WORDS}" "${MAX_MERGE_GAP_MS}" "${FALLBACK_MIN_MS}" \
  "${FALLBACK_MIN_WORDS}" "${MAX_OVERLAP_WORDS}" "${ALLOW_CROSS_SPEAKER}" <<'PY'
import argparse
import os
import sys

from c3.cli import build_correction_backend_from_args
from c3.correction import correct_details, transformer_compatible_prediction
from c3.correction_units import build_c3_predictions, summarize_unit_construction
from c3.io import load_c2_predictions, write_json, write_jsonl
from c3.schemas import CorrectionUnitConfig

(
    c2_json,
    outdir,
    correction_backend_name,
    correction_model,
    correction_api_base,
    correction_api_model,
    correction_api_key_env,
    max_new_tokens,
    min_unit_ms,
    min_unit_words,
    target_max_ms,
    hard_max_ms,
    max_unit_words,
    max_merge_gap_ms,
    fallback_min_ms,
    fallback_min_words,
    max_overlap_words,
    allow_cross_speaker,
) = sys.argv[1:]

def progress(message: str) -> None:
    print(message, flush=True)

config = CorrectionUnitConfig(
    min_unit_ms=int(min_unit_ms),
    min_unit_words=int(min_unit_words),
    target_max_ms=int(target_max_ms),
    hard_max_ms=int(hard_max_ms),
    max_unit_words=int(max_unit_words),
    max_merge_gap_ms=int(max_merge_gap_ms),
    fallback_min_ms=int(fallback_min_ms),
    fallback_min_words=int(fallback_min_words),
    max_overlap_words=int(max_overlap_words),
    respect_speaker_boundary=(allow_cross_speaker != "1"),
)

progress(f"[C3] Loading C2 predictions: {os.path.abspath(c2_json)}")
c2_predictions = load_c2_predictions(c2_json)
unit_details = build_c3_predictions(c2_predictions, config)
units_path = write_json(os.path.join(outdir, "correction_units_before_model.json"), unit_details)
units_summary_path = write_json(
    os.path.join(outdir, "correction_units_summary.json"),
    summarize_unit_construction(unit_details, c2_json, config),
)

args = argparse.Namespace(
    correction_backend=correction_backend_name,
    correction_model=correction_model,
    correction_api_base=correction_api_base,
    correction_api_model=correction_api_model,
    correction_api_key_env=correction_api_key_env,
    correction_max_new_tokens=int(max_new_tokens),
    correction_temperature=0.0,
    correction_timeout=120,
)
backend, resolved_model = build_correction_backend_from_args(args)

progress("[C3] Starting correction only")
corrected_details, prompt_records, summary = correct_details(
    unit_details,
    c2_json,
    correction_backend=backend,
    correction_model=resolved_model,
    max_new_tokens=int(max_new_tokens),
    progress_callback=progress,
)
predictions = [transformer_compatible_prediction(detail) for detail in corrected_details]

predictions_path = write_json(os.path.join(outdir, "correction_predictions.json"), predictions)
details_path = write_json(os.path.join(outdir, "correction_details.json"), corrected_details)
summary_path = write_json(os.path.join(outdir, "correction_summary.json"), summary)
prompts_path = write_jsonl(os.path.join(outdir, "prompts", "correction_prompts.jsonl"), prompt_records)

print(f"saved: {units_path}")
print(f"saved: {units_summary_path}")
print(f"saved: {predictions_path}")
print(f"saved: {details_path}")
print(f"saved: {summary_path}")
print(f"saved: {prompts_path}")
PY

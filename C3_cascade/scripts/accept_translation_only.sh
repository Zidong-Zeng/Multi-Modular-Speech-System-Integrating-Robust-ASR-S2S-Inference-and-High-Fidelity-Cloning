#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash accept_translation_only.sh --details-json CORRECTION_DETAILS_JSON --outdir OUTDIR [options]

Acceptance focus:
  C3 translation/inference model output from already corrected C3 details.

Key options:
  --details-json PATH       correction_details.json or c3_details.json.
  --outdir PATH             Output directory.
  --translation-model PATH  Local HuggingFace translation model path.
  --max-new-tokens N        Translation max_new_tokens. Default: 4096.
  --help                    Show this help.

Example:
  bash accept_translation_only.sh --details-json ../outputs/acceptance/correction/correction_details.json --outdir ../outputs/acceptance/translation
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C3_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

DETAILS_JSON="${DETAILS_JSON:-${C3_DIR}/outputs/acceptance/correction_only/correction_details.json}"
OUTDIR="${OUTDIR:-${C3_DIR}/outputs/acceptance/translation_only}"
TRANSLATION_MODEL="${TRANSLATION_MODEL:-/root/siton-tmp/assignment_C/model/Qwen2.5-1.5B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --details-json) DETAILS_JSON="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --translation-model) TRANSLATION_MODEL="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_script_logging "accept_translation_only" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"

echo "[accept_translation_only] DETAILS_JSON=${DETAILS_JSON}"
echo "[accept_translation_only] OUTDIR=${OUTDIR}"
echo "[accept_translation_only] TRANSLATION_MODEL=${TRANSLATION_MODEL}"

PYTHONPATH="${C3_DIR}/code:${PYTHONPATH:-}" python3 - "${DETAILS_JSON}" "${OUTDIR}" \
  "${TRANSLATION_MODEL}" "${MAX_NEW_TOKENS}" <<'PY'
import os
import sys

from c3.io import load_c3_details, write_json, write_jsonl
from c3.translation import translate_details, translation_prediction

details_json, outdir, translation_model, max_new_tokens = sys.argv[1:]

def progress(message: str) -> None:
    print(message, flush=True)

progress(f"[C3] Loading corrected details: {os.path.abspath(details_json)}")
details = load_c3_details(details_json)
translated_details, prompt_records, summary = translate_details(
    details,
    details_json,
    translation_model=translation_model,
    max_new_tokens=int(max_new_tokens),
    progress_callback=progress,
)
predictions = [translation_prediction(detail) for detail in translated_details]

predictions_path = write_json(os.path.join(outdir, "translation_predictions.json"), predictions)
details_path = write_json(os.path.join(outdir, "translation_details.json"), translated_details)
summary_path = write_json(os.path.join(outdir, "translation_summary.json"), summary)
prompts_path = write_jsonl(os.path.join(outdir, "prompts", "translation_prompts.jsonl"), prompt_records)

print(f"saved: {predictions_path}")
print(f"saved: {details_path}")
print(f"saved: {summary_path}")
print(f"saved: {prompts_path}")
PY

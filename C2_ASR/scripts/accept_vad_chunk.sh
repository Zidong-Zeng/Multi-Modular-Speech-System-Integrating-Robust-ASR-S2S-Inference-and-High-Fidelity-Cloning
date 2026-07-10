#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash accept_vad_chunk.sh --dataset DATASET_JSON --outdir OUTDIR [options]

Acceptance focus:
  Stage 1 VAD segmentation and Stage 2 dynamic chunk export.

Key options:
  --dataset PATH          Dataset JSON with audio/audio_path fields.
  --outdir PATH           Output directory for VAD JSON, chunk JSON, and chunk WAVs.
  --vad-backend NAME      energy | silero. Default: silero.
  --start N               Start offset. Default: 0.
  --n N                   Number of items. Default: 1.
  --threshold FLOAT       VAD threshold. Default: 0.35.
  --device NAME           Silero device: auto | cpu | cuda. Default: auto.
  --export-audio 0|1      Export chunk WAV files. Default: 1.
  --strict 0|1            Fail if selected items fail or no chunks are produced. Default: 1.
  --where KEY=VALUE       Optional dataset filter; can be repeated.
  --help                  Show this help.

Examples:
  bash accept_vad_chunk.sh --dataset ../data/cremd-test200.json --outdir ../outputs/acceptance/vad --vad-backend silero --n 1
  bash accept_vad_chunk.sh --dataset ../data/cremd-test200.json --outdir ../outputs/acceptance/vad_energy --vad-backend energy --threshold 0.2 --n 1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSIGNMENT_C_DIR="$(cd "${C2_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
source "${ASSIGNMENT_C_DIR}/scripts/logging.sh"

DATASET="${DATASET:-${C2_DIR}/data/cremd-test200.json}"
OUTDIR="${OUTDIR:-${C2_DIR}/outputs/acceptance/vad_chunk}"
VAD_BACKEND="${VAD_BACKEND:-silero}"
START="${START:-0}"
N="${N:-1}"
VAD_THRESHOLD="${VAD_THRESHOLD:-0.35}"
FRAME_MS="${FRAME_MS:-30}"
MIN_SPEECH_MS="${MIN_SPEECH_MS:-250}"
MIN_SILENCE_MS="${MIN_SILENCE_MS:-200}"
SPEECH_PAD_MS="${SPEECH_PAD_MS:-80}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
DEVICE="${DEVICE:-auto}"
MAX_CHUNK_S="${MAX_CHUNK_S:-30.0}"
MIN_CHUNK_S="${MIN_CHUNK_S:-1.0}"
MERGE_GAP_MS="${MERGE_GAP_MS:-500}"
OVERLAP_S="${OVERLAP_S:-0.5}"
EXPORT_AUDIO="${EXPORT_AUDIO:-1}"
STRICT="${STRICT:-1}"

WHERE_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --vad-backend) VAD_BACKEND="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --threshold) VAD_THRESHOLD="$2"; shift 2 ;;
    --frame-ms) FRAME_MS="$2"; shift 2 ;;
    --min-speech-ms) MIN_SPEECH_MS="$2"; shift 2 ;;
    --min-silence-ms) MIN_SILENCE_MS="$2"; shift 2 ;;
    --speech-pad-ms) SPEECH_PAD_MS="$2"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-chunk-s) MAX_CHUNK_S="$2"; shift 2 ;;
    --min-chunk-s) MIN_CHUNK_S="$2"; shift 2 ;;
    --merge-gap-ms) MERGE_GAP_MS="$2"; shift 2 ;;
    --overlap-s) OVERLAP_S="$2"; shift 2 ;;
    --export-audio) EXPORT_AUDIO="$2"; shift 2 ;;
    --strict) STRICT="$2"; shift 2 ;;
    --where) WHERE_ARGS+=(--where "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

init_script_logging "accept_vad_chunk" "${ASSIGNMENT_C_DIR}" "${ORIGINAL_ARGS[@]}"

mkdir -p "${OUTDIR}"
VAD_JSON="${OUTDIR}/stage1_${VAD_BACKEND}_vad.json"
CHUNK_JSON="${OUTDIR}/stage2_chunks.json"
CHUNK_DIR="${OUTDIR}/stage2_chunks"

echo "[accept_vad_chunk] DATASET=${DATASET}"
echo "[accept_vad_chunk] OUTDIR=${OUTDIR}"
echo "[accept_vad_chunk] VAD_BACKEND=${VAD_BACKEND}"
echo "[accept_vad_chunk] START=${START} N=${N}"
echo "[accept_vad_chunk] VAD_THRESHOLD=${VAD_THRESHOLD}"
echo "[accept_vad_chunk] CHUNK_DIR=${CHUNK_DIR}"

if [[ "${VAD_BACKEND}" == "energy" ]]; then
  stage1_cmd=(
    python3 "${C2_DIR}/code/vad_stage1_energy.py"
    --dataset "${DATASET}"
    --out "${VAD_JSON}"
    --start "${START}"
    --n "${N}"
    --threshold "${VAD_THRESHOLD}"
    --frame_ms "${FRAME_MS}"
    --min_speech_ms "${MIN_SPEECH_MS}"
    --min_silence_ms "${MIN_SILENCE_MS}"
    --speech_pad_ms "${SPEECH_PAD_MS}"
    --sample_rate "${SAMPLE_RATE}"
    "${WHERE_ARGS[@]}"
  )
elif [[ "${VAD_BACKEND}" == "silero" ]]; then
  stage1_cmd=(
    python3 "${C2_DIR}/code/vad_stage1_silero.py"
    --dataset "${DATASET}"
    --out "${VAD_JSON}"
    --start "${START}"
    --n "${N}"
    --threshold "${VAD_THRESHOLD}"
    --min_speech_ms "${MIN_SPEECH_MS}"
    --min_silence_ms "${MIN_SILENCE_MS}"
    --speech_pad_ms "${SPEECH_PAD_MS}"
    --sample_rate "${SAMPLE_RATE}"
    --device "${DEVICE}"
    "${WHERE_ARGS[@]}"
  )
else
  echo "VAD_BACKEND must be energy or silero, got: ${VAD_BACKEND}" >&2
  exit 2
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" "${stage1_cmd[@]}"

if [[ "${STRICT}" == "1" ]]; then
  python3 - "${VAD_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
summary = data.get("summary", {})
failed = int(summary.get("num_failed_items", 0) or 0)
processed = int(summary.get("num_processed_items", 0) or 0)
if failed or processed <= 0:
    print(f"[accept_vad_chunk] Stage 1 strict check failed: processed={processed}, failed={failed}", file=sys.stderr)
    for row in data.get("failures", [])[:5]:
        print(f"[accept_vad_chunk] failure: {row}", file=sys.stderr)
    sys.exit(1)
PY
fi

stage2_cmd=(
  python3 "${C2_DIR}/code/vad_stage2.py"
  --vad_json "${VAD_JSON}"
  --out "${CHUNK_JSON}"
  --chunk_dir "${CHUNK_DIR}"
  --max_chunk_s "${MAX_CHUNK_S}"
  --min_chunk_s "${MIN_CHUNK_S}"
  --merge_gap_ms "${MERGE_GAP_MS}"
  --overlap_s "${OVERLAP_S}"
)

if [[ "${EXPORT_AUDIO}" == "1" ]]; then
  stage2_cmd+=(--export_audio)
fi

PYTHONPATH="${C2_DIR}/code:${PYTHONPATH:-}" "${stage2_cmd[@]}" "${EXTRA_ARGS[@]}"

if [[ "${STRICT}" == "1" ]]; then
  python3 - "${CHUNK_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
summary = data.get("summary", {})
failed = int(summary.get("num_failed_items", 0) or 0)
processed = int(summary.get("num_processed_items", 0) or 0)
chunks = int(summary.get("num_chunks", 0) or 0)
if failed or processed <= 0 or chunks <= 0:
    print(
        f"[accept_vad_chunk] Stage 2 strict check failed: processed={processed}, failed={failed}, chunks={chunks}",
        file=sys.stderr,
    )
    for row in data.get("failures", [])[:5]:
        print(f"[accept_vad_chunk] failure: {row}", file=sys.stderr)
    sys.exit(1)
PY
fi

echo "[accept_vad_chunk] saved: ${VAD_JSON}"
echo "[accept_vad_chunk] saved: ${CHUNK_JSON}"
echo "[accept_vad_chunk] saved chunks: ${CHUNK_DIR}"

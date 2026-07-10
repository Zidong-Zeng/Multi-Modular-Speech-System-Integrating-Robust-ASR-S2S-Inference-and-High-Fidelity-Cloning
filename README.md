# Multi-Modular Speech System

**Integrating Robust ASR, Speech-to-Speech Inference, and High-Fidelity Cloning**

A multi-modular speech processing pipeline that combines robust Automatic Speech Recognition (ASR) with downstream speech-to-speech (S2S) inference and voice cloning capabilities.

---

## Project Structure

```
scripts/
  |-- logging.sh                     # Shared logging utilities
  |-- run_full_c2_c3_pipeline.sh     # End-to-end pipeline runner

C2_ASR/                              # Stage C2: Robust ASR with VAD + Diarization
  |-- code/
  |   |-- c2_asr.py                  # Main ASR entrypoint
  |   |-- c2_vad_asr.py              # VAD + ASR pipeline orchestration
  |   |-- vad_stage1_energy.py       # VAD Stage 1: Energy-based detection
  |   |-- vad_stage1_silero.py       # VAD Stage 1: Silero VAD
  |   |-- vad_stage2.py              # VAD Stage 2: Chunk segmentation
  |   |-- vad_stage3.py              # VAD Stage 3: Chunk merging
  |   |-- extract_speaker.py         # Speaker diarization (PyAnnote)
  |   |-- asr_interfaces.py          # ASR model interfaces (Whisper)
  |   |-- test-threshold/            # VAD threshold tuning tools
  |-- data/                          # Data preparation scripts
  |-- scripts/                       # Shell launchers

C3_cascade/                          # Stage C3: Cascade Correction + Translation
  |-- code/
  |   |-- c3_cascade.py              # C3 entrypoint
  |   |-- test_c3_cli_entrypoint.py  # CLI test entrypoint
  |   |-- c3/                        # C3 core module
  |       |-- pipeline.py            # Cascade pipeline orchestration
  |       |-- correction.py          # ASR error correction (LLM)
  |       |-- translation.py         # Speech translation (LLM)
  |       |-- correction_units.py    # Correction unit construction
  |       |-- text_assembly.py       # Text assembly & alignment
  |       |-- metrics.py             # Evaluation metrics
  |       |-- cli.py                 # CLI interface
  |       |-- compat.py              # Compatibility wrappers
  |       |-- io.py                  # I/O utilities
  |       |-- schemas.py             # Data schemas
  |       |-- backends/              # LLM backends (local/API)
  |       |-- prompts/               # Prompt templates
  |-- c2_code/                       # C2 ASR dependency for C3
  |-- scripts/                       # Shell launchers
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Model weights:
  - Whisper Large V3 (ASR)
  - Qwen3-4B or compatible (ASR correction)
  - Qwen2.5-1.5B-Instruct or compatible (translation)

### Run the Full Pipeline

```bash
# Set model paths
export MODEL="/path/to/whisper-large-v3"
export CORRECTION_MODEL="/path/to/Qwen3-4B"
export TRANSLATION_MODEL="/path/to/Qwen2.5-1.5B-Instruct"

# Run end-to-end
bash scripts/run_full_c2_c3_pipeline.sh
```

### Run Stages Separately

**C2: VAD + ASR only**
```bash
DATASET="/path/to/dataset.json" OUTDIR="./outputs/c2" \
  bash C2_ASR/scripts/run_c2_vad_asr.sh
```

**C3: Correction + Translation only**
```bash
C2_JSON="./outputs/c2/asr_nbest_predictions.json" OUTDIR="./outputs/c3" \
  bash C3_cascade/scripts/run_c3_cascade.sh
```

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | Whisper Large V3 path | Whisper model path |
| `ASR_MODE` | `nbest` | ASR output mode: `onebest` or `nbest` |
| `VAD_BACKEND` | `silero` | VAD backend: `energy` or `silero` |
| `ENABLE_PYANNOTE` | `0` | Enable speaker diarization |
| `CORRECTION_BACKEND` | `local` | Correction LLM: `local` or `openai_compatible` |
| `CORRECTION_MODEL` | Qwen3-4B path | Correction model path |
| `TRANSLATION_MODEL` | Qwen2.5-1.5B-Instruct path | Translation model path |

---

## Pipeline Overview

```
Audio Input
    |
    v
+-------------------------+
|  C2: VAD + ASR          |
|  +------+   +--------+  |
|  | VAD  |-->|  ASR   |  |
|  |(3-stage)| |(Whisper)| |
|  +------+   +--------+  |
|     N-best hypotheses    |
+-------------------------+
    |
    v
+-------------------------+
|  C3: Cascade            |
|  +----------+           |
|  |Correction|--> LLM    |
|  +----------+           |
|  +----------+           |
|  |Translation|--> LLM   |
|  +----------+           |
|  Corrected + Translated |
+-------------------------+
    |
    v
  Final Output
```

---

## License

This project is developed as part of the NEU NLP Lab (NiuTrans) research program.
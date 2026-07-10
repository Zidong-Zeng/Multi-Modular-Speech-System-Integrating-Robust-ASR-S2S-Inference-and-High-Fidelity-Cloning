# Multi-Modular Speech System

**Integrating Robust ASR, Speech-to-Speech Inference, and High-Fidelity Cloning**

A multi-modular speech processing pipeline that combines robust Automatic Speech Recognition (ASR) with downstream speech-to-speech (S2S) inference and voice cloning capabilities.

---

## Project Structure

```
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ logging.sh                     # Shared logging utilities
鈹?  鈹斺攢鈹€ run_full_c2_c3_pipeline.sh     # End-to-end pipeline runner
鈹溾攢鈹€ C2_ASR/                            # Stage C2: Robust ASR with VAD + Diarization
鈹?  鈹溾攢鈹€ code/
鈹?  鈹?  鈹溾攢鈹€ c2_asr.py                  # Main ASR entrypoint
鈹?  鈹?  鈹溾攢鈹€ c2_vad_asr.py              # VAD + ASR pipeline orchestration
鈹?  鈹?  鈹溾攢鈹€ vad_stage1_energy.py       # VAD Stage 1: Energy-based detection
鈹?  鈹?  鈹溾攢鈹€ vad_stage1_silero.py       # VAD Stage 1: Silero VAD
鈹?  鈹?  鈹溾攢鈹€ vad_stage2.py              # VAD Stage 2: Chunk segmentation
鈹?  鈹?  鈹溾攢鈹€ vad_stage3.py              # VAD Stage 3: Chunk merging
鈹?  鈹?  鈹溾攢鈹€ extract_speaker.py         # Speaker diarization (PyAnnote)
鈹?  鈹?  鈹溾攢鈹€ asr_interfaces.py          # ASR model interfaces (Whisper)
鈹?  鈹?  鈹斺攢鈹€ test-threshold/            # VAD threshold tuning tools
鈹?  鈹溾攢鈹€ data/                          # Data preparation scripts
鈹?  鈹斺攢鈹€ scripts/                       # Shell launchers
鈹溾攢鈹€ C3_cascade/                        # Stage C3: Cascade Correction + Translation
鈹?  鈹溾攢鈹€ code/
鈹?  鈹?  鈹溾攢鈹€ c3_cascade.py              # C3 entrypoint
鈹?  鈹?  鈹斺攢鈹€ c3/                        # C3 core module
鈹?  鈹?      鈹溾攢鈹€ pipeline.py            # Cascade pipeline orchestration
鈹?  鈹?      鈹溾攢鈹€ correction.py          # ASR error correction (LLM)
鈹?  鈹?      鈹溾攢鈹€ translation.py         # Speech translation (LLM)
鈹?  鈹?      鈹溾攢鈹€ correction_units.py    # Correction unit construction
鈹?  鈹?      鈹溾攢鈹€ text_assembly.py       # Text assembly & alignment
鈹?  鈹?      鈹溾攢鈹€ metrics.py             # Evaluation metrics
鈹?  鈹?      鈹溾攢鈹€ cli.py                 # CLI interface
鈹?  鈹?      鈹溾攢鈹€ backends/              # LLM backends (local/API)
鈹?  鈹?      鈹斺攢鈹€ prompts/               # Prompt templates
鈹?  鈹溾攢鈹€ c2_code/                       # C2 ASR dependency for C3
鈹?  鈹斺攢鈹€ scripts/                       # Shell launchers
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
| `MODEL` | `/root/siton-tmp/.../whisper-large-v3` | Whisper model path |
| `ASR_MODE` | `nbest` | ASR output mode: `onebest` or `nbest` |
| `VAD_BACKEND` | `silero` | VAD backend: `energy` or `silero` |
| `ENABLE_PYANNOTE` | `0` | Enable speaker diarization |
| `CORRECTION_BACKEND` | `local` | Correction LLM: `local` or `openai_compatible` |
| `CORRECTION_MODEL` | `/root/siton-tmp/.../Qwen3-4B` | Correction model path |
| `TRANSLATION_MODEL` | `/root/siton-tmp/.../Qwen2.5-1.5B-Instruct` | Translation model path |

---

## Pipeline Overview

```
Audio Input
    鈹?    鈻?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? C2: VAD + ASR          鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹?  鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹? 鈹?VAD  鈹傗攢鈹€鈻垛攤  ASR   鈹? 鈹?鈹? 鈹?3-stage)鈹?鈹?Whisper)鈹? 鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹?  鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?鈹?      N-best hypotheses  鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹?    鈻?鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹? C3: Cascade            鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?          鈹?鈹? 鈹侰orrection鈹傗攢鈹€鈻?LLM    鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?          鈹?鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?          鈹?鈹? 鈹俆ranslation鈹傗攢鈹€鈻?LLM   鈹?鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?          鈹?鈹?   Corrected + Translated鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?    鈹?    鈻?  Final Output
```

---

## License

This project is developed as part of the NEU NLP Lab (NiuTrans) research program.

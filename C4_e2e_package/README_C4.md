# C4 code package

This package contains C4 executable code only. It does not include model weights, generated audio, full result JSON files, or other intermediate outputs.

## Main script for the integrated system

```bash
conda activate speech_tcx
cd /root/siton-tmp/assignment_C
python C4_end2end/code/c4_e2e.py \
  --dataset common_data/dataset.json \
  --model model/seamless-m4t-v2-large \
  --outdir C4_end2end/outputs_core \
  --batch_size 50
```

## C1 interface

Use C1 manifest if available:

```bash
python C4_end2end/code/c4_e2e.py \
  --dataset C1_audio_processing/sample_data/sample_manifest.json \
  --model model/seamless-m4t-v2-large \
  --outdir C4_end2end/outputs_core_c1
```

If the manifest is empty or not generated, scan a C1 audio folder directly:

```bash
python C4_end2end/code/c4_e2e.py \
  --input_dir C1_audio_processing/sample_data/augmented/clean \
  --model model/seamless-m4t-v2-large \
  --outdir C4_end2end/outputs_core_c1_clean
```

## Output for C5

Read `<outdir>/c4_results.json`. Important fields:

- `id`
- `audio`
- `audio_abs`
- `emotion`
- `speaker`
- `e2e_translation_zh`

C5 should use `e2e_translation_zh` as the Chinese text input.

## Advanced scripts included

- `test_kimi_audio_s2s_twostage.py`: Kimi-Audio S2S two-stage smoke/advanced experiment.
- `qwen25_omni_dataset.py`: Qwen2.5-Omni S2T/S2S, emotion prompt, rerank, reference-audio experiments.
- `qwen25_omni_multiturn.py`: audio-only multi-turn dialogue experiment.
- `seamless_multiturn_baseline.py`: SeamlessM4T single-turn baseline for multi-turn comparison.
- `evaluate_emotion_preservation.py`, `extract_prosody_features.py`, `prosody_utils.py`, `train_emotion_controller.py`: emotion/prosody evaluation and controller experiments.
- `compare_qwen_seamless.py`: Qwen vs Seamless comparison utility.

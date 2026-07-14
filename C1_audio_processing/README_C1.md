# C1 语音数据处理与音频增强

This module is the first part of the multi-modular speech system, responsible for providing unified and standardized audio input for the entire speech system.

## Environment setup

The core dependencies are as follows:librosa, soundfile, torchaudio, pytorch, matplotlib, ffmpeg
```bash
conda create -n your_env python=3.10 -y
conda activate your_env
pip install -r requirements.txt
```

## Main script for audio processing (基础任务)

Process a single audio file and generate waveform, spectrogram, and report.

```bash
cd C1_audio_processing/code
python c1_audio_processing.py --input ../data/sample.wav --outdir C1_audio_processing/outputs
```

## C1 Advanced augmentation (进阶增强)

Generate VAD segments, speed/volume perturbations, and noise-added audio.

```bash
python c1_advanced_augmentation.py \
  --input common_data/dataset.json \
  --outdir C1_audio_processing/outputs/augmented \
  --target_sr 16000
```

Available flags:
- `--noise_file <path>`: add background noise with specific SNR (5/10/15dB).
- `--no-vad`: disable VAD segmentation (recommended for short audios).
- `--no-speed`: disable speed perturbation.
- `--no-volume`: disable volume perturbation.
- `--no-noise`: disable noise augmentation.

Plot augmented audio: Scan the augmented folder and generate 4-in-1 overview plots (Waveform + Spectrogram + Mel + MFCC) for each `.wav`.

```bash
python c1_plot_augmented_folder.py \
  --input C1_audio_processing/outputs/augmented \
  --outdir C1_audio_processing/outputs/plots_augmented \
```


## Multi-sr comparison (采样率对比)

Analyze the effect of different sample rates (8000Hz vs 44100Hz).

```bash
python c1_dif_resample_rate.py \
  --input common_data/dataset.json \
  --outdir C1_audio_processing/outputs/multi_sr_compare
```

## Output file structure

When executed, the following structure will be generated under `<outdir>`:

```text
augmented/                               # 增强输出根目录
├── clean/                               # 基础处理后的干净语音
│   └── <id>_clean.wav
├── vad/                                 # VAD 切割短句片段
│   └── <id>_vad_seg{i}.wav
├── speed/                               # 速度扰动 (0.9 / 1.1 倍)
│   └── <id>_speed_{0.9/1.1}.wav
├── volume/                              # 音量扰动 (0.5 / 1.5 倍)
│   └── <id>_volume_{0.5/1.5}.wav
├── noise/                               # 噪声增强 (SNR 5/10/15)
│   └── <id>_noise_snr{5/10/15}.wav
└── augmented_manifest.json              # 流水线元数据清单（包含绝对路径）
plot_augmented/                          # 增强输出根目录
├── clean/                               
│   └── *_overview.png                   # 波形、FBank、MFCC 四合一图
├── vad/                                 
│   └── *_overview.png
├── speed/                               # 同上
├── volume/                              
├── noise/                               
multi_sr_compare/
├── sr_8000/                   # 8kHz采样率结果目录
│   ├── *_overview.png         # 波形、FBank、MFCC 四合一图
│   └── C1_report.txt          # 详细处理日志与统计信息
└── sr_44100/                  # 44.1kHz采样率结果目录
```

## Output for C2, C3, C4

C2/C3/C4 可以使用 `augmented_manifest.json` 作为 `--dataset` 参数，对增强后的音频进行 ASR 或级联推理测试。还用提取数据样本的脚本设置了sample_data数据集，其中放置了短音频和拼接后的长音频，可以用作对比测试。

## Core acoustic concepts

- **STFT**: 时频变换，生成线性频谱。
- **FBank**: 通过 Mel 滤波器组 + log 运算获得，符合人耳听觉特性。
- **MFCC**: FBank 进行 DCT 得到，去相关后的关键声学特征。
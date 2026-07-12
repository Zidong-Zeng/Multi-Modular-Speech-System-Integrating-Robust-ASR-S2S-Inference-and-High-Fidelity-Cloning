#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


def add_repo(repo: str):
    p = Path(repo).expanduser().resolve()
    if not (p / 'kimia_infer').exists():
        raise FileNotFoundError(f'kimia_infer not found under {p}')
    sys.path.insert(0, str(p))
    print(f'[INFO] Added repo: {p}')


def resolve_audio_path(dataset_path: Path, item: dict) -> Path:
    raw = item.get('audio') or item.get('audio_path')
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        dataset_path.parent / p,
        dataset_path.parent / 'dataset' / 'cremad' / p,
        dataset_path.parent / 'dataset' / 'cremad' / 'AudioWAV' / p.name,
        Path('/root/siton-tmp/assignment_C/common_data/dataset/cremad/AudioWAV') / p.name,
        Path('/root/siton-tmp/assignment_C/C2_ASR/data/dataset/cremad/AudioWAV') / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(raw)


def load_sample(dataset_path: Path, index: int):
    data = json.loads(dataset_path.read_text(encoding='utf-8'))
    item = data[index]
    return item, resolve_audio_path(dataset_path, item)


@torch.inference_mode()
def detokenize_audio(detokenizer, audio_tokens: torch.Tensor) -> torch.Tensor:
    detokenizer.clear_states()
    chunk_size = 30
    first_chunk_size = 30
    cache_speech_collection = []
    audio_tokens = audio_tokens.to(torch.cuda.current_device()).long()
    num_audio_tokens = audio_tokens.size(1)
    first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
    gen_speech = detokenizer.detokenize_streaming(
        first_chunk_semantic_tokens,
        is_final=(num_audio_tokens <= first_chunk_size),
        upsample_factor=4,
    )
    cache_speech_collection.append(gen_speech)
    if num_audio_tokens > first_chunk_size:
        res_semantic_tokens = audio_tokens[:, first_chunk_size:]
        for i in range(0, res_semantic_tokens.size(1), chunk_size):
            chunk_semantic_tokens = res_semantic_tokens[:, i:i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_semantic_tokens,
                upsample_factor=4,
                is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
            )
            cache_speech_collection.append(gen_speech)
    return torch.cat(cache_speech_collection, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../model/Kimi-Audio-7B-Instruct')
    parser.add_argument('--dataset', default='../../common_data/dataset.json')
    parser.add_argument('--kimi_repo', default='../../third_party/Kimi-Audio')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=300)
    parser.add_argument('--out_wav', default=None, help='Output wav path. Default: ../outputs/kimi_s2s_<sample_id>.wav')
    parser.add_argument('--out_json', default=None, help='Output json path. Default: ../outputs/kimi_s2s_<sample_id>.json')
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    model_path = (here / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    dataset_path = (here / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    repo_path = (here / args.kimi_repo).resolve() if not Path(args.kimi_repo).is_absolute() else Path(args.kimi_repo)
    print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
    print(f'[INFO] model={model_path}')
    print(f'[INFO] dataset={dataset_path}')
    add_repo(str(repo_path))

    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.models.detokenizer import get_audio_detokenizer

    item, audio_path = load_sample(dataset_path, args.index)
    sample_id = str(item.get('id') or f'index_{args.index:04d}')
    safe_sample_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in sample_id)
    out_wav_arg = args.out_wav or f'../outputs/kimi_s2s_{safe_sample_id}.wav'
    out_json_arg = args.out_json or f'../outputs/kimi_s2s_{safe_sample_id}.json'
    out_wav = (here / out_wav_arg).resolve() if not Path(out_wav_arg).is_absolute() else Path(out_wav_arg)
    out_json = (here / out_json_arg).resolve() if not Path(out_json_arg).is_absolute() else Path(out_json_arg)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] sample index={args.index} id={sample_id} audio={audio_path}')
    print(f'[INFO] output wav={out_wav}')
    print(f'[INFO] output json={out_json}')

    messages = [
        {'role': 'user', 'message_type': 'text', 'content': '请严格执行语音翻译任务。你将听到一段英文语音。请只生成中文语音，内容必须是该英文语音的中文翻译。不要输出英文，不要重复原文，不要解释。如果原语音带有情绪，请用相同情绪的中文语气朗读。'},
        {'role': 'user', 'message_type': 'audio', 'content': str(audio_path)},
    ]
    sampling = dict(
        audio_temperature=0.8,
        audio_top_k=10,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
    )

    print('[INFO] Stage 1: load main model without detokenizer')
    t0 = time.time()
    model = KimiAudio(model_path=str(model_path), load_detokenizer=False)
    print(f'[INFO] main model loaded in {time.time() - t0:.1f}s')

    history = model.prompt_manager.get_prompt(messages, output_type='both')
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    audio_features = history.continuous_feature
    audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
    text_input_ids = text_input_ids.to(torch.cuda.current_device())
    is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
    audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

    print('[INFO] Stage 1: generate text/audio tokens')
    t1 = time.time()
    wav_tokens, text_tokens = model._generate_loop(
        audio_input_ids=audio_input_ids,
        text_input_ids=text_input_ids,
        max_new_tokens=args.max_new_tokens,
        is_continuous_mask=is_continuous_mask,
        continous_feature=audio_features,
        output_type='both',
        **sampling,
    )
    gen_sec = time.time() - t1
    wav_tokens = [t for t in wav_tokens if t >= model.kimia_token_offset]
    wav_tokens = torch.tensor(wav_tokens).unsqueeze(0) - model.kimia_token_offset
    text_tokens = [t for t in text_tokens if t < model.kimia_token_offset]
    text_output = model.detokenize_text(text_tokens)
    print(f'[RESULT] text_output: {text_output}')
    print(f'[INFO] generated audio tokens: {wav_tokens.numel()} in {gen_sec:.1f}s')

    kimia_token_offset = model.kimia_token_offset
    del audio_input_ids, text_input_ids, is_continuous_mask, audio_features, history
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f'[INFO] GPU memory after freeing main model: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB')

    if wav_tokens.numel() == 0:
        raise RuntimeError('No audio tokens generated; try increasing --max_new_tokens')

    print('[INFO] Stage 2: load detokenizer and synthesize wav')
    t2 = time.time()
    detokenizer = get_audio_detokenizer(str(model_path))
    wav = detokenize_audio(detokenizer, wav_tokens)
    synth_sec = time.time() - t2
    sf.write(str(out_wav), wav.detach().cpu().view(-1).numpy(), 24000)

    result = {
        'id': item.get('id'),
        'input_audio': str(audio_path),
        'label_text': item.get('text'),
        'label_emotion': item.get('emotion'),
        'text_output': text_output,
        'audio_output': str(out_wav),
        'num_audio_tokens': int(wav_tokens.numel()),
        'generate_sec': round(gen_sec, 3),
        'synthesize_sec': round(synth_sec, 3),
        'max_new_tokens': args.max_new_tokens,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[RESULT] audio_output: {out_wav}')
    print(f'[RESULT] json_output: {out_json}')
    print(f'[INFO] synthesize time: {synth_sec:.1f}s')


if __name__ == '__main__':
    main()

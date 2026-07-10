# -*- coding: utf-8 -*-
"""
灏嗘寚瀹氱洰褰曚笅鐨勫涓?.flac 鐭煶棰戯紝鎸夋枃浠跺悕椤哄簭鎷兼帴鎴愪竴涓暱鐨?.wav 闊抽銆?
鐢ㄦ硶绀轰緥:
    python merge_flac_to_wav.py \
        --input_dir /root/siton-tmp/assignment_C/C2_ASR/data/librispeech/121/123852 \
        --output /root/siton-tmp/assignment_C/C2_ASR/data/librispeech/121/123852/121-123852_merged.wav

渚濊禆: soundfile, numpy
    pip install soundfile numpy --break-system-packages   # 濡傛湭瀹夎
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import soundfile as sf


def natural_sort_key(path: str):
    """鎸夋枃浠跺悕涓殑鏁板瓧鑷劧鎺掑簭锛岄伩鍏?0009 鎺掑湪 00010 鍚庨潰杩欑闂銆?""
    name = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def load_flac_files(input_dir: str) -> list[str]:
    files = glob.glob(os.path.join(input_dir, "*.flac"))
    if not files:
        raise FileNotFoundError(f"鐩綍涓嬫病鏈夋壘鍒?.flac 鏂囦欢: {input_dir}")
    files.sort(key=natural_sort_key)
    return files


def merge_flac_to_wav(input_dir: str, output_path: str, target_sr: int | None = None) -> None:
    files = load_flac_files(input_dir)
    print(f"鍏辨壘鍒?{len(files)} 涓?flac 鏂囦欢锛屾寜椤哄簭鍚堝苟:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    audio_chunks = []
    sr = None

    for f in files:
        data, file_sr = sf.read(f, dtype="float32", always_2d=False)
        # 澶氬０閬撹浆鍗曞０閬?        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr is None:
            sr = file_sr
        elif file_sr != sr:
            raise ValueError(
                f"閲囨牱鐜囦笉涓€鑷? {f} 鏄?{file_sr}Hz锛屼箣鍓嶇殑鏂囦欢鏄?{sr}Hz锛?
                "璇峰厛缁熶竴閲囨牱鐜囧啀鍚堝苟锛堟垨鍛婅瘔鎴戦渶瑕佽嚜鍔ㄩ噸閲囨牱锛夈€?
            )

        audio_chunks.append(data)

    merged = np.concatenate(audio_chunks, axis=0)

    out_sr = target_sr or sr
    if target_sr and target_sr != sr:
        # 濡傞渶閲嶉噰鏍凤紝鐢?librosa锛堝彲閫変緷璧栵級
        import librosa

        merged = librosa.resample(merged, orig_sr=sr, target_sr=target_sr)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, merged, out_sr, subtype="PCM_16")

    duration_s = len(merged) / out_sr
    print(f"\n鍚堝苟瀹屾垚: {output_path}")
    print(f"閲囨牱鐜? {out_sr} Hz, 鏃堕暱: {duration_s:.2f}s ({duration_s/60:.2f} min)")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="鍚堝苟鐩綍涓嬬殑flac闊抽涓轰竴涓獁av闀块煶棰?)
    ap.add_argument(
        "--input_dir",
        default="/root/siton-tmp/assignment_C/C2_ASR/data/librispeech/121/121726",
        help="鍖呭惈澶氫釜 .flac 鏂囦欢鐨勭洰褰?,
    )
    ap.add_argument(
        "--output",
        default="/root/siton-tmp/assignment_C/C2_ASR/data/libris-merge/121-121726_merged.wav",
        help="杈撳嚭鐨勫悎骞秝av鏂囦欢璺緞",
    )
    ap.add_argument(
        "--target_sr",
        type=int,
        default=None,
        help="鍙€夛細閲嶉噰鏍风洰鏍囬噰鏍风巼锛堜笉濉垯淇濇寔鍘熼噰鏍风巼锛學hisper甯哥敤16000锛?,
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    merge_flac_to_wav(args.input_dir, args.output, target_sr=args.target_sr)


if __name__ == "__main__":
    main()
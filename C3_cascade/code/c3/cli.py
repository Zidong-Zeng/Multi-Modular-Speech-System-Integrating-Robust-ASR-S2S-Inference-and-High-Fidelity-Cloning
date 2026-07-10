# -*- coding: utf-8 -*-
"""Command line entrypoint for the modular C3 cascade."""

from __future__ import annotations

import argparse
import os

from .backends.openai_compatible import OpenAICompatibleCorrectionBackend
from .pipeline import run_cascade
from .schemas import CorrectionUnitConfig


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="C3 modular cascade runner")
    ap.add_argument("--c2_json", default="/root/siton-tmp/assignment_C/C3_cascade/outputs/c2-nbest/cremd200-tiny/test.json", help="C2 asr_nbest_predictions.json")
    ap.add_argument("--correction_backend", choices=["local", "openai_compatible"], default="local")
    ap.add_argument("--correction_model", default="/root/siton-tmp/assignment_C/model/Qwen3-4B",help="Local HuggingFace correction model path")
    ap.add_argument("--correction_api_base", default="https://api.deepseek.com",help="OpenAI-compatible API base URL, for example https://api.example.com/v1")
    ap.add_argument("--correction_api_key_env", default="CORRECTION_API_KEY", help="Environment variable containing the correction API key")
    ap.add_argument("--correction_api_model", default="deepseek-v4-pro",help="OpenAI-compatible correction model name")
    ap.add_argument("--correction_temperature", type=float, default=0.0)
    ap.add_argument("--correction_timeout", type=int, default=120)
    ap.add_argument("--translation_model", default="/root/siton-tmp/assignment_C/model/Qwen2.5-1.5B-Instruct", help="Local HuggingFace translation model path")
    ap.add_argument("--correction_max_new_tokens", type=int, default=4096)
    ap.add_argument("--translation_max_new_tokens", type=int, default=4096)
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C3_cascade/outputs/c3-corrected/cremd200-tiny-qw3")
    ap.add_argument("--min_unit_ms", type=int, default=6000)
    ap.add_argument("--min_unit_words", type=int, default=10)
    ap.add_argument("--target_max_ms", type=int, default=25000)
    ap.add_argument("--hard_max_ms", type=int, default=29500)
    ap.add_argument("--max_unit_words", type=int, default=80)
    ap.add_argument("--max_merge_gap_ms", type=int, default=1200)
    ap.add_argument("--fallback_min_ms", type=int, default=0,help="閽堝闀块煶棰戯紝璇ュ弬鏁伴槇鍊间负2950")
    ap.add_argument("--fallback_min_words", type=int, default=0,help="閽堝闀块煶棰戯紝璇ュ弬鏁伴槇鍊间负5")
    ap.add_argument("--max_overlap_words", type=int, default=8)
    ap.add_argument("--allow_cross_speaker", action="store_true", help="Allow one correction unit to span speakers")
    return ap

def build_correction_backend_from_args(args, parser: argparse.ArgumentParser | None = None):
    def fail(message: str):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if args.correction_backend == "local":
        if not args.correction_model:
            fail("--correction_model is required when --correction_backend local")
        return None, args.correction_model

    if args.correction_backend == "openai_compatible":
        if not args.correction_api_base:
            fail("--correction_api_base is required when --correction_backend openai_compatible")
        if not args.correction_api_model:
            fail("--correction_api_model is required when --correction_backend openai_compatible")
        backend = OpenAICompatibleCorrectionBackend.from_env(
            api_base=args.correction_api_base,
            api_key_env=args.correction_api_key_env,
            model=args.correction_api_model,
            max_new_tokens=args.correction_max_new_tokens,
            temperature=args.correction_temperature,
            timeout=args.correction_timeout,
        )
        return backend, args.correction_api_model

    fail(f"unsupported correction backend: {args.correction_backend}")


def config_from_args(args) -> CorrectionUnitConfig:
    return CorrectionUnitConfig(
        min_unit_ms=args.min_unit_ms,
        min_unit_words=args.min_unit_words,
        target_max_ms=args.target_max_ms,
        hard_max_ms=args.hard_max_ms,
        max_unit_words=args.max_unit_words,
        max_merge_gap_ms=args.max_merge_gap_ms,
        fallback_min_ms=args.fallback_min_ms,
        fallback_min_words=args.fallback_min_words,
        max_overlap_words=args.max_overlap_words,
        respect_speaker_boundary=not args.allow_cross_speaker,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    correction_backend, correction_model = build_correction_backend_from_args(args, parser)

    def print_progress(message: str) -> None:
        print(message, flush=True)

    result = run_cascade(
        args.c2_json,
        args.outdir,
        correction_backend=correction_backend,
        correction_model=correction_model,
        translation_model=args.translation_model,
        correction_max_new_tokens=args.correction_max_new_tokens,
        translation_max_new_tokens=args.translation_max_new_tokens,
        config=config_from_args(args),
        progress_callback=print_progress,
    )
    print(f"saved: {result.predictions_path}")
    print(f"saved: {result.details_path}")
    print(f"saved: {result.summary_path}")
    print(f"saved: {result.correction_prompts_path}")
    print(f"saved: {result.translation_prompts_path}")


if __name__ == "__main__":
    main()

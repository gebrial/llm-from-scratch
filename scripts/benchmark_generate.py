"""
Time ModelService.generate() on a few fixed prompts, to compare generation speed
before and after a change (e.g. adding a KV cache).

Seeds are fixed per prompt, so a change that keeps the outputs identical also
generates the same number of tokens -- the timings are then directly comparable.

Usage:
    python scripts/benchmark_generate.py checkpoints/run1/epoch=1-step=7042-inference.ckpt
"""
import argparse
import time

import torch

import _pathsetup  # noqa: F401
from model_service import ModelService

PROMPTS = [
    "Once upon a time",
    "One day a girl walked into",
    "Tom had a red ball. He",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    # the instance is a t3.small with 2 vCPUs
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--method", default="top_p")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    service = ModelService(args.checkpoint, args.tokenizer)

    total_tokens, total_seconds = 0, 0.0
    for seed, prompt in enumerate(PROMPTS):
        torch.manual_seed(seed)
        start = time.perf_counter()
        story = service.generate(prompt, method=args.method)
        seconds = time.perf_counter() - start

        num_tokens = len(service.tokenizer.encode(story).ids) - len(service.tokenizer.encode(prompt).ids)
        total_tokens += num_tokens
        total_seconds += seconds
        print(f"{prompt!r}: {num_tokens} tokens in {seconds:.1f}s ({num_tokens / seconds:.1f} tok/s)")
        print(f"    {story[:120]!r}...")

    print(f"total: {total_tokens} tokens in {total_seconds:.1f}s ({total_tokens / total_seconds:.1f} tok/s)")


if __name__ == "__main__":
    main()

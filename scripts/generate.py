"""
Generate story completions from a trained checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/run/epoch=1-step=7044.ckpt \
        --tokenizer data/tokenizer.json --prompt "One day a girl walked into"

    # batch mode: generate a completion for every prompt in a CSV with a "prompt" column
    python scripts/generate.py --checkpoint ... --tokenizer ... \
        --prompts-csv evaluation_prompts.csv --output completions.csv
"""
import argparse
import csv

import _pathsetup  # noqa: F401 -- adds src/ to sys.path before importing components

import torch
import tokenizers.decoders
from tokenizers import Tokenizer

from components.generatetext import beam_search, generate_text, top_p_sampling
from train import LitGPTModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--method", choices=["top_p", "beam", "greedy"], default="top_p")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=3, help="Used by --method greedy")
    parser.add_argument("--beams", type=int, default=5, help="Used by --method beam")

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="A single prompt to complete")
    prompt_group.add_argument("--prompts-csv", help="CSV file with a 'prompt' column")
    parser.add_argument("--output", help="Where to write --prompts-csv results (CSV)")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    litmodel = LitGPTModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
    litmodel.eval()
    litmodel.model.to(device)
    return litmodel.model


def complete(model, tokenizer, prompt, args):
    if args.method == "top_p":
        return top_p_sampling(model, tokenizer, prompt, top_p=args.top_p,
                               temperature=args.temperature, max_length=args.max_length)
    if args.method == "beam":
        beams = beam_search(model, tokenizer, prompt, max_beams=args.beams, max_tokens=args.max_length)
        return beams[0] if beams else ""
    device = next(model.parameters()).device
    return generate_text(model, tokenizer, prompt, args.max_length, device,
                          temperature=args.temperature, topk=args.topk, output_only=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    model = load_model(args.checkpoint, device)

    if args.prompt:
        print(complete(model, tokenizer, args.prompt, args))
        return

    with open(args.prompts_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=1):
        row["completion"] = complete(model, tokenizer, row["prompt"], args)
        if i % 10 == 0:
            print(f"Generated {i}/{len(rows)}")

    output_path = args.output or "completions.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} completions to {output_path}")


if __name__ == "__main__":
    main()

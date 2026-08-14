"""
Evaluate a trained checkpoint's loss, accuracy, and perplexity on the packed dataset.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/run1/epoch=1-step=....ckpt \
        --dataset data/packed_dataset
"""
import argparse
import math

import _pathsetup  # noqa: F401 -- adds src/ to sys.path before importing components

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader

from train import LitGPTModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/packed_dataset")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    litmodel = LitGPTModel.load_from_checkpoint(checkpoint_path=args.checkpoint, map_location=device)
    litmodel.eval()
    litmodel.to(device)

    dataset = load_from_disk(args.dataset)[args.split]
    dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, y = litmodel._step(batch)

            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=-1) == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"checkpoint={args.checkpoint} split={args.split} tokens={total_tokens}")
    print(f"loss={avg_loss:.4f} accuracy={accuracy:.4f} perplexity={perplexity:.2f}")


if __name__ == "__main__":
    main()

"""
Strip a training checkpoint down to what inference needs.

A Lightning checkpoint holds everything needed to resume training, and for this
model two thirds of it is Adam's optimizer state (two buffers per parameter).
Serving never reads that, but load_from_checkpoint still loads all of it into
memory, which is enough to OOM a 2 GB instance. The stripped file is still a
Lightning checkpoint, so LitGPTModel.load_from_checkpoint loads it unchanged.

Usage:
    python scripts/strip_checkpoint.py --checkpoint "checkpoints/run1/epoch=1-step=7042.ckpt"
    # writes checkpoints/run1/epoch=1-step=7042-inference.ckpt
"""
import argparse
from pathlib import Path

import torch

# What load_from_checkpoint reads: the weights, the config to rebuild the model
# with, and the version it uses to migrate older checkpoint formats. epoch and
# global_step are kept only so the file still says which run it came from.
INFERENCE_KEYS = [
    "state_dict",
    "hyper_parameters",
    "hparams_name",
    "pytorch-lightning_version",
    "epoch",
    "global_step",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="Defaults to <checkpoint>-inference.ckpt alongside it")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path.with_name(
        f"{checkpoint_path.stem}-inference.ckpt"
    )

    # weights_only=False: hyper_parameters is a plain dict, but Lightning
    # checkpoints can hold arbitrary pickled objects, and this is our own file.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stripped = {key: checkpoint[key] for key in INFERENCE_KEYS if key in checkpoint}
    dropped = sorted(set(checkpoint) - set(stripped))

    torch.save(stripped, output_path)

    print(f"dropped: {', '.join(dropped)}")
    print(f"{checkpoint_path.stat().st_size / 1e6:.0f} MB -> {output_path.stat().st_size / 1e6:.0f} MB")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

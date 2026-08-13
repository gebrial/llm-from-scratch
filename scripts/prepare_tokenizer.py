"""
Train a byte-level BPE tokenizer on the TinyStories dataset.

Usage:
    python scripts/prepare_tokenizer.py --output data/tokenizer.json
"""
import argparse

from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<|endoftext|>", "\n", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tokenizer.json",
                         help="Path to write the trained tokenizer to")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=418,
                         help="Drop stories shorter than this (roughly the shortest 1%%)")
    parser.add_argument("--max-chars", type=int, default=2505,
                         help="Drop stories longer than this (roughly the longest 1%%)")
    return parser.parse_args()


def is_clean_story(example, min_chars, max_chars):
    text = example["text"]
    if len(text) < min_chars or len(text) > max_chars:
        return False
    return all(ord(char) < 128 for char in text)


def main():
    args = parse_args()

    dataset = load_dataset("roneneldan/TinyStories")
    dataset = dataset.filter(lambda ex: is_clean_story(ex, args.min_chars, args.max_chars))

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        vocab_size=args.vocab_size,
        show_progress=True,
    )

    def batch_iterator(batch_size=1000):
        train_split = dataset["train"]
        for i in range(0, len(train_split), batch_size):
            yield train_split[i:i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.save(args.output)
    print(f"Saved tokenizer (vocab_size={tokenizer.get_vocab_size()}) to {args.output}")


if __name__ == "__main__":
    main()
